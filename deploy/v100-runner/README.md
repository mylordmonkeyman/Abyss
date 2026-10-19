# Docker runner for the V100 server

For Ubuntu Server 24.04, the existing NVIDIA driver, V100 PCIe 32 GB (host GPU 0),
two Xeon E5-2697A v4 CPUs and 128 GB RAM. This setup adds a repository runner for
`mylordmonkeyman/Abyss`; it does not reconfigure the existing NInfer runner.

The image supplies its own CUDA 12.8 compiler/libraries. The host NVIDIA driver
and NVIDIA Container Toolkit provide GPU access. The host CUDA installation is
not mounted over the container's toolkit.

## 1. Verify Docker and GPU access

On the server:

```bash
nvidia-smi
sudo docker version
sudo docker compose version
sudo docker run --rm --gpus device=0 \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 nvidia-smi
```

The final command should show the V100. If it works, continue to step 2. Existing
GPU-enabled NInfer containers usually mean the container runtime is already set up.

If Docker is installed but GPU access fails because the NVIDIA Container Toolkit
is missing, follow NVIDIA's installation procedure:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Restarting Docker can interrupt other containers; perform that last step with
those workloads stopped. Then repeat the GPU test. Do not reinstall the NVIDIA
driver or CUDA on a host where GPU containers already work.

If `docker compose version` fails, install the Compose v2 plugin appropriate to
your Docker installation (for Docker's apt repository: `sudo apt-get install docker-compose-plugin`).

## 2. Get the runner files

Use a separate checkout for runner administration:

```bash
mkdir -p ~/ai
cd ~/ai
git clone --branch feature/deepseek-v100-perf \
  https://github.com/mylordmonkeyman/Abyss.git abyss-runner-setup
cd ~/ai/abyss-runner-setup/deploy/v100-runner
```

If this directory already exists, update it instead of cloning again:

```bash
cd ~/ai/abyss-runner-setup
git pull --ff-only
cd deploy/v100-runner
```

## 3. Set the runner version and image checksum

Open:

https://github.com/mylordmonkeyman/Abyss/settings/actions/runners/new

Choose **Linux**, **x64**. In the download commands, note:

- The version in `actions-runner-linux-x64-VERSION.tar.gz` (without the leading `v`).
- The SHA-256 checksum in GitHub's checksum-validation command.

Do not run those host installation commands; the Docker build handles installation.
Enter the displayed version/checksum:

```bash
read -r -p 'GitHub runner version (without v): ' ABYSS_RUNNER_VERSION
read -r -p 'Linux-x64 runner SHA256: ' ABYSS_RUNNER_SHA256
ABYSS_GPU_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
cat > .env <<EOF_ENV
ABYSS_RUNNER_VERSION=$ABYSS_RUNNER_VERSION
ABYSS_RUNNER_SHA256=$ABYSS_RUNNER_SHA256
ABYSS_GPU_UUID=$ABYSS_GPU_UUID
EOF_ENV
chmod 600 .env
sudo docker compose config --quiet
sudo docker compose build --pull
```

The `.env` contains configuration, not a registration token. The build verifies
the runner download against the GitHub-provided checksum. It can take several minutes.

## 4. Register once with Abyss

On the same GitHub page, get the short-lived token from the `./config.sh --url ...
--token ...` command. Generate a fresh token if the image build took over an hour.
Enter it at the prompt, then run registration:

```bash
read -r -s -p 'Abyss runner registration token: ' ABYSS_RUNNER_TOKEN
printf '\n'
sudo docker compose run --rm --no-deps runner \
  ./config.sh --unattended \
  --url https://github.com/mylordmonkeyman/Abyss \
  --token "$ABYSS_RUNNER_TOKEN" \
  --name abyss-v100-sm70 \
  --labels v100,sm70,cuda-12-8 \
  --work _work
unset ABYSS_RUNNER_TOKEN
```

Use the **Abyss registration token**, not a personal access token and not a token
from the NInfer repository. GitHub adds `self-hosted`, `Linux`, `X64` automatically.
Runner configuration is retained in `abyss-v100-runner-data`; no token is needed on
subsequent starts. Do not use `--replace` for an existing runner unless deliberately
replacing its registration.

## 5. Start and verify

```bash
sudo docker compose up -d --no-build
sudo docker compose logs --tail 100 -f runner
```

Look for `Connected to GitHub` and `Listening for Jobs`. Ctrl+C exits log viewing
without stopping the container. Check that **abyss-v100-sm70** appears online at:

https://github.com/mylordmonkeyman/Abyss/settings/actions/runners

Verify tools:

```bash
sudo docker compose exec runner bash -lc '
  /usr/local/cuda-12.8/bin/nvcc --version
  /usr/local/cuda-12.8/bin/cuobjdump --version
  /usr/local/cuda-12.8/bin/compute-sanitizer --version
  cmake --version
  ninja --version
  nvidia-smi
'
```

The workflow uses GPU 0, the V100, and compiles with 32 jobs. Its default variables
match this image; no repository-variable changes are needed. GPU jobs are restricted
to owner `mylordmonkeyman` pushes/manual runs on main or the V100 feature branch;
other actors' runs are skipped before assignment to the runner. See [security controls](SECURITY.md).

## 6. Start the workflow

Check https://github.com/mylordmonkeyman/Abyss/actions for the queued
**V100 CUDA 12.8 build and device smoke** job from the workflow commit. If still
queued, it should be picked up once this runner is online.

To trigger a new run without merging the feature branch, push a workflow-only
comment change:

```bash
cd ~/ai/abyss-runner-setup
printf '\n# Trigger V100 runner bring-up.\n' >> .github/workflows/v100.yml
git add .github/workflows/v100.yml
git commit -m 'ci: trigger V100 runner bring-up'
git push origin feature/deepseek-v100-perf
```

These Git commands require your usual GitHub push authentication. Alternatively,
use GitHub to rerun an existing V100 workflow run. The manual **Run workflow** UI
is available once this workflow exists on the repository's default branch.

The workflow builds the engine, checks code-object images, launches a small device
test and runs memcheck/synccheck. Logs are uploaded as `abyss-v100-<run>-<attempt>`.
It does not download or run the full DeepSeek model. Do not run concurrent GPU tests
or inference on the NInfer runner/local model server during measured Abyss testing;
GitHub concurrency does not coordinate separate repositories.

## Maintenance and persistence

From `~/ai/abyss-runner-setup/deploy/v100-runner`:

```bash
sudo docker compose ps
sudo docker compose logs --tail 200 runner
sudo docker compose stop
sudo docker compose start
```

`restart: unless-stopped` starts the container after host/Docker restart unless
you explicitly stopped it. The named volume retains the runner, auto-updates,
checkout, build cache and temporary files. No host model directory is mounted by
default. To deliberately expose a dedicated model-only directory, set
`ABYSS_MODEL_DIR` in `.env` and use the optional `compose.models.yml` override described
in [SECURITY.md](SECURITY.md). Read-only mounts still allow every job to read/copy
those files. No host Docker socket, privileged mode or published port is needed.

Use `docker compose down` to remove the container while keeping the named volume.
**Do not use `down -v` or delete `abyss-v100-runner-data`** when preserving the runner
registration and builds. Logs are size-limited. No snapshot/commit of the container
is required. If changing the image's runner version later, the existing volume keeps
its installed, auto-updated runner; rebuilding does not overwrite its registration.
