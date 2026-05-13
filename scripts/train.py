import multiprocessing
from omegaconf import OmegaConf
from datetime import datetime
import logging, torch, argparse, os, wandb, time
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import matplotlib, glob

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edengnn.data.dataload import get_loader
from edengnn.data.io_vasp import IO_VASP
from edengnn.data.io_openmx import IO_OpenMX
from edengnn.model.model import EfficientDensity

colors = plt.cm.tab10.colors


class Model(L.LightningModule):
    """-------------------------------------------------------------------------

    I mask the augmentation occupancy after model output, because it is more
    convenient to recover chgcar format with irreps form than with compact form.

    -------------------------------------------------------------------------"""

    def __init__(
        self,
        model: torch.nn.Module,
        wandb_run,
        logger,
        save_dir,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.wandb_run = wandb_run
        self.logger_ = logger
        self.save_dir = save_dir

        self.lr = kwargs.get("lr", 1e-2)
        self.max_steps = kwargs.get("max_steps", 1000000)

        self.loss_fn = {
            "L1": torch.nn.L1Loss(),
            "L2": torch.nn.MSELoss(),
        }

        self.log_every_n_steps = kwargs.get("log_every_n_steps", 1)
        self.check_val_every_n_epoch = kwargs.get("check_val_every_n_epoch", 1)
        self.log_level = kwargs.get("log_level", 0)

        self.val_check_interval_steps = kwargs.get("val_check_interval_steps", None)
        self.optimizer = kwargs.get("optimizer", None)
        self.scheduler = kwargs.get("scheduler", None)
        self.loss_dict = kwargs.get("loss_dict", None)
        self.freeze_basis = kwargs.get("freeze_basis", False)
        self.task = kwargs.get("task", 2)

        self.best_val = torch.tensor(float("inf"))
        # for parity plot of augmentation occupancy
        self.preds_aug = []
        self.targets_aug = []
        self.predictions = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = 0
        for key, value in output.items():
            if key == "aug_tensor":
                loss += (
                    self.loss_fn[self.loss_dict[key]["loss"]](
                        batch[key].flatten()[batch["aug_mask"]],
                        value.flatten()[batch["aug_mask"]],
                    )
                    * self.loss_dict[key]["weight_train"]
                )

            elif key == "total_charge":
                loss += (
                    self.loss_fn[self.loss_dict[key]["loss"]](
                        batch["grid_func_out"].mean(), value
                    )
                    * self.loss_dict[key]["weight_train"]
                )
            else:
                loss += (
                    self.loss_fn[self.loss_dict[key]["loss"]](batch[key], value)
                    * self.loss_dict[key]["weight_train"]
                )

        if self.global_step % self.log_every_n_steps == 0:
            self.wandb_run.log(
                {
                    "train/loss": loss.item(),
                    "step": self.global_step,
                    "time": time.time(),
                }
            )
            lr = self.optimizers().param_groups[0]["lr"]
            if self.log_level == 0:
                self.logger_.info(
                    f"[Train] Step {self.global_step}, Loss: {loss.item():.4f}, lr: {lr}"
                )
            elif self.log_level == 1:
                self.logger_.info(
                    f"[Train] Step {self.global_step}, Loss: {loss.item():.4f}, lr: {lr}, \
                    name: {batch["name"]}, number of atoms: {batch["nat"]}, number of points: {batch["npb"]}"
                )

        return loss

    def validation_step(self, batch, batch_idx):

        output = self(batch)
        losses = {}
        for key, value in output.items():
            if key == "grid_func_out":
                losses[key] = torch.sum(torch.abs(batch[key] - value)) / torch.sum(
                    batch[key]
                )
            elif key == "aug_tensor":
                aug_tar = batch[key].flatten()[batch["aug_mask"]]
                aug_pre = value.flatten()[batch["aug_mask"]]
                losses[key] = self.loss_fn["L1"](aug_tar, aug_pre)
                self.targets_aug.append(aug_tar.detach().cpu())
                self.preds_aug.append(aug_pre.detach().cpu())
            elif key == "total_charge":
                losses[key] = (
                    self.loss_fn["L1"](value, batch["grid_func_out"].mean())
                    * batch["volume"][0]
                )

        info_log = f"[Validation] Epoch {self.current_epoch + 1}, "
        loss = 0
        for key, value in losses.items():
            info_log += f"{key} Loss: {value},"
            loss += losses[key] * self.loss_dict[key]["weight_val"]
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.ptr.size(0) - 1,
            sync_dist=True,
        )

        self.log_dict(
            {f"val/{k}_loss": v.detach() for k, v in losses.items()},
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.ptr.size(0) - 1,
            sync_dist=True,
        )
        self.logger_.info(info_log)

    def on_validation_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]

        val_metrics = self.trainer.callback_metrics

        info_log = f"[Validation End] Epoch {self.current_epoch + 1}, "
        wandb_dict = {
            "train/lr": lr,
            "epoch": self.current_epoch + 1,
            "time": time.time(),
        }
        for k, v in val_metrics.items():
            if k.startswith("val"):
                value = v.item()
                wandb_dict[k] = value
                info_log += f"{k}: {value},"

        self.wandb_run.log(wandb_dict)
        self.logger_.info(info_log)

        self._plot_aug()
        #
        # self.lr_schedulers().step(val_metrics["val/loss"])

        current_val = float(val_metrics["val/loss"])
        if self.current_epoch > 0:
            if current_val < self.best_val:
                try:
                    os.remove(glob.glob(os.path.join(self.save_dir, "best_*.ckpt"))[0])
                except:
                    pass
                self.best_val = current_val
                self.trainer.save_checkpoint(
                    filepath=os.path.join(
                        self.save_dir, f"best_step={self.global_step}.ckpt"
                    )
                )
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        losses = {}
        for key, value in output.items():
            if key == "grid_func_out":
                losses[key] = torch.sum(torch.abs(batch[key] - value)) / torch.sum(
                    batch[key]
                )
            elif key == "aug_tensor":
                aug_tar = batch[key].flatten()[batch["aug_mask"]]
                aug_pre = value.flatten()[batch["aug_mask"]]
                losses[key] = self.loss_fn["L1"](aug_tar, aug_pre)
                self.targets_aug.append(aug_tar.detach().cpu())
                self.preds_aug.append(aug_pre.detach().cpu())
            elif key == "total_charge":
                losses[key] = (
                    self.loss_fn["L1"](value, batch["grid_func_out"].mean())
                    * batch["volume"]
                )

        info_log = f"[Test] {batch["name"]}, "
        loss = 0
        for key, value in losses.items():
            info_log += f"{key} Loss: {value},"
            loss += losses[key] * self.loss_dict[key]["weight_val"]
        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.ptr.size(0) - 1,
            sync_dist=True,
        )

        self.log_dict(
            {f"test/{k}_loss": v.detach() for k, v in losses.items()},
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.ptr.size(0) - 1,
            sync_dist=True,
        )
        self.logger_.info(info_log)

    def on_test_end(self):
        test_metrics = self.trainer.callback_metrics

        info_log = f"[Test End], "
        wandb_dict = {
            "time": time.time(),
        }
        for k, v in test_metrics.items():
            if k.startswith("test"):
                value = v.item()
                wandb_dict[k] = value
                info_log += f"{k}: {value},"

        self.wandb_run.log(wandb_dict)
        self.logger_.info(info_log)
        self._plot_aug()

    def predict_step(self, batch, batch_idx):
        output = self(batch)
        self.logger_.info(f"[Predict], name: {batch["name"]}")
        result = {
            "name": batch["name"][0],
            "z": batch["z"].detach().cpu().numpy(),
            "pos": batch["pos"].detach().cpu().numpy(),
            "cell": batch["cell"].detach().cpu().numpy(),
            "nat": batch["nat"].detach().cpu().numpy(),
            "npb": batch["npb_total"].detach().cpu().numpy(),
            "volume": batch["volume"].detach().cpu().numpy()[0],
        }
        if "lmix_max" in batch:
            result["lmix_max"] = batch["lmix_max"].detach().cpu().numpy()[0]
        if self.task == 0:
            result["density"] = output["grid_func_out"].detach().cpu().numpy()
            result["aug"] = None
        else:
            result["aug"] = output["aug_tensor"].detach().cpu().numpy()
            if self.task == 1:
                result["density"] = None
            else:
                result["density"] = output["grid_func_out"].detach().cpu().numpy()
        return result

    def configure_optimizers(self):
        if self.optimizer.type == "AdamW":
            if self.freeze_basis:
                self.logger_.info("[Setting] freezing basis enabled")
                self.model.probe.radial_basis_probe.requires_grad_(False)
                self.model.probe.z_net.requires_grad_(False)
                self.model.probe.edge_net.requires_grad_(False)
            if self.task == 0:
                self.model.aug.requires_grad_(False)
            elif self.task == 1:
                self.model.probe.requires_grad_(False)

            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.lr,
                eps=self.optimizer.AdamW.epsilon,
                betas=(self.optimizer.AdamW.beta1, self.optimizer.AdamW.beta2),
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        if self.scheduler.type == "ReduceLROnPlateau":
            if self.val_check_interval_steps:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=self.scheduler.ReduceLROnPlateau.factor,
                        patience=self.scheduler.ReduceLROnPlateau.patience,
                        min_lr=self.scheduler.ReduceLROnPlateau.min_lr,
                    ),
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": self.val_check_interval_steps,
                    "strict": True,
                }
            else:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=self.scheduler.ReduceLROnPlateau.factor,
                        patience=self.scheduler.ReduceLROnPlateau.patience,
                        min_lr=self.scheduler.ReduceLROnPlateau.min_lr,
                    ),
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.check_val_every_n_epoch,
                    "strict": True,
                }

        return [optimizer], [scheduler]

    def _plot_aug(self):
        if len(self.targets_aug) > 0:
            preds = torch.cat(self.preds_aug)
            targets = torch.cat(self.targets_aug)

            plt.figure()
            plt.plot(
                [targets.min(), targets.max()],
                [targets.min(), targets.max()],
                color=colors[1],
            )
            plt.scatter(targets, preds, s=2, color=colors[0], zorder=10)
            plt.xlabel("True")
            plt.ylabel("Pred")
            plt.title("Parity Plot--Aug")
            img_path = os.path.join(self.save_dir, "parity_aug_val.png")
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.targets_aug = []
            self.preds_aug = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(cfg.run.save_dir, cfg.run.project, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_dir, "info.log"),
        filemode="w",
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    device = cfg.run.device
    DTYPE = (
        torch.float32
        if cfg.run.dtype in ["float32", "torch.float32"]
        else torch.float64
    )
    torch.set_default_dtype(DTYPE)
    torch.set_float32_matmul_precision(
        "medium" if torch.get_default_dtype() == torch.float32 else "highest"
    )

    logger.info("[Init] Building model...")
    model = EfficientDensity(cfg.model).to(DTYPE)
    logger.info("%s", OmegaConf.to_yaml(cfg.model))

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        dir=save_dir,
        mode=cfg.wandb.mode,
        config=cfg,
    )
    wandb_run.watch(model, log="all")

    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=30, verbose=True, mode="min"
    )

    num_gpus = cfg.run.get("num_gpus", 1)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in device else "cpu",
        devices=num_gpus,
        strategy=(
            DDPStrategy(gradient_as_bucket_view=True) if num_gpus > 1 else "auto"
        ),
        logger=False,
        default_root_dir=save_dir,
        log_every_n_steps=cfg.optimize.get("log_every_n_steps", 10),
        check_val_every_n_epoch=cfg.optimize.get("check_val_every_n_epoch", 1),
        val_check_interval=cfg.optimize.get("val_check_interval_steps", None),
        gradient_clip_val=cfg.optimize.get("grad_clip", None),
        enable_progress_bar=False,
        accumulate_grad_batches=cfg.optimize.get("accumulate_grad_batches", 1),
        callbacks=[early_stop_callback],
        max_epochs=cfg.optimize.get("max_epochs", 3000),
        max_steps=cfg.optimize.get("max_steps", 1000000),
    )

    if cfg.data.dft_software == "openmx":
        io_dft = IO_OpenMX(
            stage=cfg.run.mode,
            save_dir=save_dir,
            path_template=cfg.data.openmx.path_template,
            encut=cfg.data.openmx.encut,
            num_proc=cfg.data.openmx.num_proc,
            dk_bz=cfg.data.openmx.dk_bz,
            dk_band=cfg.data.openmx.dk_band,
            plot_band=cfg.data.openmx.plot_band,
        )
    else:
        io_dft = IO_VASP(
            stage=cfg.run.mode,
            save_dir=save_dir,
            dir=cfg.data.dir,
            use_bin=cfg.data.use_bin,
            path_template=cfg.data.vasp.path_template,
            encut=cfg.data.vasp.encut,
            lmix_max=cfg.data.vasp.lmix_max,
            dk_bz=cfg.data.vasp.dk_bz,
            dk_band=cfg.data.vasp.dk_band,
            plot_band=cfg.data.vasp.plot_band,
        )

    if cfg.run.mode == "train":
        train_loader = get_loader(cfg, stage="train", io_dft=io_dft)
        val_loader = get_loader(cfg, stage="val", io_dft=io_dft)
        if cfg.run.resume:
            lightning_model = Model(
                model=model,
                wandb_run=wandb_run,
                logger=logger,
                save_dir=save_dir,
                **cfg.optimize,
            )
            trainer.fit(
                lightning_model,
                train_loader,
                val_loader,
                ckpt_path=cfg.run.get("checkpoint", None),
            )
        elif cfg.run.checkpoint and not cfg.run.resume:
            lightning_model = Model.load_from_checkpoint(
                cfg.run.checkpoint,
                model=model,
                wandb_run=wandb_run,
                logger=logger,
                save_dir=save_dir,
                **cfg.optimize,
            )
            trainer.fit(lightning_model, train_loader, val_loader)
        elif not cfg.run.checkpoint and not cfg.run.resume:
            logger.info("[Init] Train from scratch")
            logger.info("[Init] Data\n%s", OmegaConf.to_yaml(cfg.data))
            logger.info("[Init] Optimize\n%s", OmegaConf.to_yaml(cfg.optimize))
            lightning_model = Model(
                model=model,
                wandb_run=wandb_run,
                logger=logger,
                save_dir=save_dir,
                **cfg.optimize,
            )
            trainer.fit(lightning_model, train_loader, val_loader)
    else:
        lightning_model = Model.load_from_checkpoint(
            cfg.run.checkpoint,
            model=model,
            wandb_run=wandb_run,
            logger=logger,
            save_dir=save_dir,
            **cfg.optimize,
        )
        if cfg.run.mode == "test":
            loader = get_loader(cfg, stage="test", io_dft=io_dft)
            trainer.test(lightning_model, loader)

        elif cfg.run.mode == "predict":
            loader = get_loader(cfg, stage="predict", io_dft=io_dft)

            t_predict_start = time.time()
            predictions = trainer.predict(lightning_model, loader)
            t_predict_total = time.time() - t_predict_start

            num_atom_total = 0
            num_pb_total = 0
            MAX_PROCESSES = min(os.cpu_count(), 8) - 1

            logger.info(f"[Write]: using {MAX_PROCESSES} cores......")
            t_write_start = time.time()

            tasks = []
            if cfg.data.dft_software == "vasp":
                for struct in predictions:
                    task_args = (
                        struct["name"],
                        struct["aug"],
                        struct["density"],
                        struct["z"],
                        struct["pos"],
                        struct["cell"],
                        struct["volume"],
                    )
                    tasks.append(task_args)
                    num_atom_total += struct["nat"]
                    num_pb_total += struct["npb"]

                with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
                    pool.starmap(io_dft.write_density, tasks)
            elif cfg.data.dft_software == "openmx":
                for struct in predictions:
                    task_args = (
                        struct["name"],
                        struct["density"],
                    )
                    tasks.append(task_args)
                    num_atom_total += struct["nat"]
                    num_pb_total += struct["npb"]

                with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
                    pool.starmap(io_dft.write_density, tasks)

            num_atom_total = num_atom_total[0]
            num_pb_total = num_pb_total[0]
            t_write_total = time.time() - t_write_start

            logger.info(
                "[Stat]: "
                + f"\n\taverage predict speed: {(num_atom_total/t_predict_total):.1f} atoms/s"
                + f"\n\taverage predict speed: {(num_pb_total/t_predict_total):.0f} grids/s"
                + f"\n\taverage write speed: {(num_atom_total/t_write_total):.1f} atoms/s"
            )


if __name__ == "__main__":
    main()
