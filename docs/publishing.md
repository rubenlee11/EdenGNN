# Build and publish this documentation

This documentation is a static site built with MkDocs Material. Markdown files in `docs/` are its content source; `mkdocs.yml` defines navigation, search, theme, and the GitHub project URL. Building the site does not require EdenGNN, CUDA, a DFT executable, or model downloads.

## Preview locally

From the repository root, install the small, separate documentation dependency set. For example, create an environment outside the source tree:

```bash
python -m venv /tmp/edengnn-docs-env
source /tmp/edengnn-docs-env/bin/activate
python -m pip install -r requirements-docs.txt
mkdocs serve --dev-addr 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/`. Pages rebuild when their Markdown source changes. The theme includes navigation, a page table of contents, full-text search, code-copy buttons, and light/dark modes.

For a production build:

```bash
mkdocs build --strict --site-dir /tmp/edengnn-docs-site
```

The generated HTML is under `/tmp/edengnn-docs-site/`. The default configuration also writes outside the source checkout. Use `--site-dir` or the `EDENGNN_DOCS_SITE_DIR` environment variable to choose another output directory. Generated HTML and build environments do not belong in the source repository.

## Publish with GitHub Pages

The included `.github/workflows/docs.yml` builds the site and deploys a Pages artifact. It uses GitHub's artifact deployment instead of a generated `gh-pages` branch.

1. Push the documentation files, `mkdocs.yml`, `requirements-docs.txt`, and the workflow to the repository's `main` branch.
2. Open the GitHub repository's **Settings → Pages**. Under **Build and deployment → Source**, select **GitHub Actions**.
3. In **Actions → Documentation**, run the workflow manually if the initial push happened before Pages was enabled.
4. Wait for the build and deployment jobs to succeed. The deployment job reports the live URL.

For `rubenlee11/EdenGNN`, the configured project-site URL is:

```text
https://rubenlee11.github.io/EdenGNN/
```

This is the expected URL after a successful deployment, not a claim that Pages is already enabled. Pull requests build the documentation for validation; they do not deploy it. Relevant changes pushed to `main` deploy automatically, and manual workflow dispatch is also supported.

For a fork or a renamed repository, update `site_url`, `repo_url`, and `repo_name` in `mkdocs.yml`. If the default branch differs from `main`, also update the workflow branch filter, deployment condition, and `edit_uri`.

## Edit the content

Edit the corresponding Markdown file and add new pages to `nav` in `mkdocs.yml`. Use relative links between documentation pages so they work in both GitHub source views and the generated site. Run a strict build before publishing to catch missing page targets and configuration problems.

Keep computation outputs, model weights, credentials, local pseudopotentials, and machine-specific job results outside `docs/`.
