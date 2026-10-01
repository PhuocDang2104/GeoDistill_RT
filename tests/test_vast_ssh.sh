#!/usr/bin/env bash
# Run only in an isolated Ubuntu container with openssh-server/client installed.
set -euo pipefail
[[ "${GEODISTILL_DISPOSABLE_SSH_TEST:-}" == "1" ]] || exit 2
mkdir -p /run/sshd /root/.ssh
ssh-keygen -q -t ed25519 -N '' -f /tmp/accepted
ssh-keygen -q -t ed25519 -N '' -f /tmp/rejected
cp /tmp/accepted.pub /root/.ssh/authorized_keys
# Reproduce rejection of a correct key when provider-created permissions are bad.
chmod 777 /root/.ssh
chmod 666 /root/.ssh/authorized_keys
ssh-keygen -A
/usr/sbin/sshd -E /tmp/auth-before.log
ssh_opts=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
if ssh "${ssh_opts[@]}" -i /tmp/accepted root@localhost true; then
  echo 'Expected bad-permissions rejection' >&2; exit 1
fi
grep 'bad ownership or modes' /tmp/auth-before.log
sha256sum /root/.ssh/authorized_keys /etc/ssh/ssh_host_*_key > /tmp/original-sha
bash /repo/docker/vast-ssh.sh
ssh "${ssh_opts[@]}" -i /tmp/accepted root@localhost 'echo LOGIN_OK'
if ssh "${ssh_opts[@]}" -i /tmp/rejected root@localhost true; then
  echo 'Unexpected unauthorized key acceptance' >&2; exit 1
fi
sha256sum -c /tmp/original-sha
bash /repo/docker/vast-ssh.sh
ssh "${ssh_opts[@]}" -i /tmp/accepted root@localhost 'echo REPEAT_LOGIN_OK'
sha256sum -c /tmp/original-sha
echo 'PASS: permissions repaired, actual SSH login, wrong-key rejection, key preservation, repeat startup'
