# Build and publish the documentation

The website uses MkDocs Material. Content is stored in `docs/`; navigation and theme settings are in `mkdocs.yml`.

## Preview and build

Install the dependencies in `requirements-docs.txt` in a separate Python environment. Use MkDocs `serve` for local preview or `build` with strict checking to generate the static website.

The default output directory is `/tmp/edengnn-docs-site/`. Set `EDENGNN_DOCS_SITE_DIR` or the `--site-dir` option to choose another location. Keep environments and generated HTML outside the source checkout.

## Publish to GitHub Pages

1. Push the documentation and build configuration to `main`.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Open the [Documentation workflow](https://github.com/rubenlee11/EdenGNN/actions/workflows/docs.yml) and select **Run workflow**. Alternatively, rerun an existing Documentation job.
4. Wait for the build and deployment jobs to succeed.

The configured website is [rubenlee11.github.io/EdenGNN](https://rubenlee11.github.io/EdenGNN/). Documentation changes pushed to `main` deploy automatically. Pull requests check the build without deployment.

## Update pages

Edit the Markdown files and register new pages in `mkdocs.yml`. Use relative links between pages and check a strict build before publishing.

For a renamed repository or fork, update the repository/site URLs in `mkdocs.yml`. Update the branch settings in the configuration and workflow if the default branch differs from `main`.
