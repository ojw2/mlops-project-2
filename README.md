# Project 2: CI/CD for ML Model Deployment

Builds on project 1. Every push now flows through an automated pipeline:

```
push/PR ──> lint + tests (incl. model accuracy gate)
        ──> build image ──> deploy to throwaway kind cluster IN CI ──> smoke test live endpoint
        ──> (main branch only) push versioned image to GitHub Container Registry
```

Then you pull the published image into your local cluster — no more building on your laptop.

## What's new vs project 1

```
├── .github/workflows/ci-cd.yaml   # the pipeline
├── tests/
│   ├── test_train.py              # artifact contract + accuracy quality gate
│   └── test_api.py                # endpoint tests via FastAPI TestClient
├── k8s/
│   ├── base/                      # deployment + service (from project 1)
│   └── overlays/
│       ├── local/                 # local dev tag
│       └── prod/                  # GHCR image; CI sets the tag
├── pyproject.toml                 # pytest + ruff config
├── requirements-dev.txt
└── Makefile                       # make test / lint / serve / build
```

Key idea: **k8s manifests are now Kustomize-managed.** The base never mentions a registry or tag; overlays inject them. CI runs `kustomize edit set image` instead of sed-ing YAML.

## Setup

1. Create a GitHub repo and push this code:
   ```bash
   git init && git add -A && git commit -m "project 2: ci/cd"
   git remote add origin git@github.com:YOUR_USERNAME/iris-mlops.git
   git push -u origin main
   ```
2. Edit `k8s/overlays/prod/kustomization.yaml` — replace `OWNER` with your GitHub username.
3. That's it. `GITHUB_TOKEN` is provided automatically; the workflow's `packages: write` permission lets it push to GHCR. No secrets to configure.

> If your GHCR package ends up private (default for personal accounts), your local cluster will need an `imagePullSecret`, or just make the package public in the package settings — fine for a learning project.

## Watch the pipeline work

Push any change and open the **Actions** tab. The three jobs:

**1. `test`** — ruff lint, then trains the model and runs pytest. Note `test_model_meets_accuracy_floor`: this is an *ML quality gate*. If a code change degrades the model below 85% accuracy, the pipeline blocks the deploy. Try it — set `n_estimators=1` and `max_depth=1` in `train.py` and watch CI fail.

**2. `integration`** — builds the Docker image, boots a **kind cluster inside the CI runner**, loads the image, deploys with kustomize, waits for rollout, then curls the live `/predict` endpoint through the Service. This catches the class of bugs unit tests can't: bad probes, missing files in the image, port mismatches, resource limits too tight to start.

**3. `publish`** — only on pushes to `main` (never PRs). Pushes the image to GHCR tagged with the git SHA plus `latest`, so every image is traceable to an exact commit.

## Deploy a published image locally

Once `publish` has run:

```bash
# point the prod overlay at a specific SHA tag (pin, don't use latest)
cd k8s/overlays/prod
kustomize edit set image iris-classifier=ghcr.io/YOUR_USERNAME/iris-classifier:<short-sha>
kubectl apply -k .
kubectl rollout status deployment/iris-classifier
```

Your cluster now pulls the exact image CI tested — the same artifact, not a rebuild. That's the core CD principle: **build once, promote the same artifact everywhere.**

## Exercises

1. **Break the accuracy gate.** Cripple the model in `train.py`, open a PR, watch `test` fail and block merge. Add branch protection requiring the checks to pass.
2. **Break the container, not the code.** Change the Dockerfile `EXPOSE`/CMD port to 9000 without touching the k8s manifests. Unit tests pass; the `integration` job fails at rollout. This demonstrates why the kind-in-CI stage exists.
3. **Trace an image to a commit.** Pick any image tag in GHCR, find the commit, `git checkout` it, and confirm what code produced that artifact.
4. **Tagged releases.** `git tag v0.2.0 && git push --tags` — the workflow publishes a `v0.2.0` image. Deploy it via the prod overlay.
5. **PR flow.** Open a PR and confirm `test` + `integration` run but `publish` doesn't. Why does that matter? (Untested images should never reach the registry.)

## Concepts you just learned

- **Quality gates for models, not just code** — accuracy floors as CI assertions; the beginning of continuous training discipline.
- **Ephemeral environments in CI** — kind gives you a real k8s API server per pipeline run, free.
- **Build once, promote everywhere** — SHA-tagged immutable images; `latest` is for convenience, deploys pin SHAs.
- **Kustomize overlays** — one base, many environments, no YAML duplication or sed.

## Gap to notice (foreshadowing project 5)

The "CD" here still ends with you running `kubectl apply` by hand, and the model is retrained inside every image build. Real pipelines separate *model* releases from *code* releases and pull models from a registry. That's exactly what projects 4–5 fix — MLflow registry, then pipeline orchestration.

## Next: Project 3

Real model serving — replace the hand-rolled FastAPI deployment with KServe, and add autoscaling driven by request load.
