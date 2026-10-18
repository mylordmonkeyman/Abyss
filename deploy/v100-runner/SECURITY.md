# Public repository / self-hosted runner security

The public source repository does not give an anonymous visitor access to this
runner by itself. This GPU workflow has no pull-request/comment/workflow-run trigger.
It is intended to execute code you control and have reviewed. GitHub nevertheless
recommends private repositories for self-hosted runners because untrusted code can
compromise persistent runners:
https://docs.github.com/en/actions/reference/security/secure-use

## Controls in these files

- GitHub evaluates a job-level condition **before scheduling**: exact Abyss repository,
  owner `mylordmonkeyman` as both original and rerun actor, push/manual event only,
  main or the V100 feature branch only. A non-owner GitHub App/bot identity is not
  implicitly allowed. Check the event's actor if an intended automated push is skipped.
- The workflow checks out the triggered commit, with no arbitrary ref/PR input.
  Permissions remain `contents: read`; credentials are not persisted in checkouts.
  No repository secret/PAT is passed into the job. Runner registration credentials
  still exist inside the runner's persistent volume.
- Checkout/upload actions are pinned to full commit SHAs verified against their
  official repositories. The GGML checkout and runner download checksum are pinned.
  This is pinning, not a claim that every transitive dependency/image package was audited.
- The image runs as UID 1001. Compose drops all Linux capabilities, enables
  `no-new-privileges`, limits CPU/memory/PIDs, exposes only the selected GPU, and uses
  Docker's default isolation. There is no Docker socket, privileged mode, host network,
  SSH/home-directory mount, or default model mount. Do not add SYS_ADMIN or make this
  runner privileged to work around a profiler permission problem.
- CODEOWNERS routes changes to workflows/runner setup to the owner. It only becomes
  an enforced review requirement when matching GitHub branch rules are enabled.

The condition protects this workflow as written. It cannot protect against someone
with permission to rewrite it, add another GPU workflow, or bypass branch rules.
Labels such as `v100` select a runner; they are not an authorization boundary.
Other current workflows use GitHub-hosted runners. A fork PR can propose changing
one of those workflows to target this runner; the condition in `v100.yml` does NOT
constrain that different job. Requiring approval for all outside collaborators and
reviewing every workflow change matters even though `v100.yml` has no PR trigger.
Keep untrusted PR CI on GitHub-hosted runners. Branch-merge review is not the same
as controlling which pre-merge PR workflows may run.

## GitHub settings to apply yourself

These settings were NOT changed by the code commit:

1. **Settings > Actions > General**: keep default workflow permissions read-only;
   disable the option allowing Actions to create/approve pull requests. Require approval
   for **all outside collaborators** under fork-PR workflow approvals. Approval is not
   a sandbox: do not approve an unreviewed PR workflow targeting this self-hosted runner.
2. **Settings > Rules > Rulesets** (or branch protection): protect `main` and
   `feature/deepseek-v100-perf` against unauthorized updates/deletion/force-pushes;
   require owner review of workflow/runner setup changes when accepting collaborators'
   PRs. Configure owner exceptions deliberately so your own authorized work can continue.
   CODEOWNERS on a feature branch does not enforce rules by itself; put it on the
   relevant protected base branch when using PR review enforcement.
3. Keep repository write/admin access limited. Do not place broad PATs, personal SSH
   keys, cloud credentials or legal/client files in runner volumes, environment variables,
   model mounts or the image. Review artifact/log contents before deliberately adding
   sensitive inputs; public workflow artifacts can expose readable job outputs.

If using an organization runner group that supports workflow restrictions, restrict
both the repository and specific workflow/ref. This is not a setting available on
this personal-repository runner, so do not assume an owner-only YAML condition is
an equivalent platform-level restriction.

## Apply these changes to an existing container

Wait until the runner is idle, then run on the server:

```bash
cd ~/ai/abyss-runner-setup
git pull --ff-only
cd deploy/v100-runner
sudo docker compose config --quiet
sudo docker compose up -d --no-build --force-recreate
sudo docker compose logs --tail 100 runner
```

This recreates the container with the hardened Compose settings and removes the
previous default host-model mount. The named registration/build volume is retained;
no re-registration is needed. If your earlier workflow-trigger comment commit is
unpublished and makes `git pull --ff-only` fail, inspect/reconcile that local commit
rather than resetting or deleting the runner volume.

## Optional model access

For intentional trusted model runs, create a dedicated directory containing only
model assets that jobs may read, set `ABYSS_MODEL_DIR=/path/to/that/directory` in `.env`,
and start with:

```bash
sudo docker compose -f compose.yml -f compose.models.yml up -d --no-build --force-recreate
```

Read-only protects files against writes, **not disclosure**. Never point this mount
at your home directory or any directory containing private/client material. Omit the
override again to remove model access. The current build/device smoke workflow needs
no models and should use the default configuration.

## Limits and stronger isolation

This remains a **persistent, trusted-code** runner. A malicious build can modify its
workspace, caches, installed runner files or registration credentials as UID 1001;
retaining the volume also retains that compromise. A container restart/clean checkout
is not recovery. If compromised, revoke/remove its GitHub runner registration, preserve
needed evidence, rebuild the environment from trusted inputs, and use a fresh volume.
Rotate any credentials that the compromised environment could read.

Docker GPU access shares the host kernel/NVIDIA driver, and ordinary bridge networking
can reach host/LAN services. Capability drops do not eliminate those attack surfaces.
For sensitive networks, isolate the CI host/network and explicitly restrict outbound
access to internal services using host/network firewall policy; these files do not
install host-wide firewall rules. Resource limits bound memory/PIDs but not growth
of the persistent volume; monitor free space.

For a stronger public/private boundary, keep Abyss source public but register the GPU
runner with a **separate private CI repository**. That repository's trusted workflow
checks out explicitly reviewed public Abyss SHAs. Keep registration tokens and CI
control in the private repository. This separates public contribution events from
runner authorization; it still requires reviewing the code you execute. For actually
untrusted builds, use disposable isolated machines/VMs with a clean disk per job and
appropriate GPU isolation, rather than this persistent development container.
