#!/usr/bin/env bash
set -euo pipefail

# Preserve provider-injected keys and authentication policy. Only repair file
# ownership/modes, missing host keys, and make authentication failures observable.
mkdir -p /run/sshd /root/.ssh
chown root:root /root /root/.ssh
chmod go-w /root
chmod 700 /root/.ssh
for keyfile in /root/.ssh/authorized_keys /root/.ssh/authorized_keys2; do
  if [[ -f "$keyfile" ]]; then
    chown root:root "$keyfile"
    chmod 600 "$keyfile"
    printf 'SSH authorized-key fingerprints: %s\n' "$keyfile"
    ssh-keygen -lf "$keyfile" | awk '{print $1, $2, $NF}' || true
  fi
done
ssh-keygen -A
/usr/sbin/sshd -t
echo 'SSH effective configuration for root:'
/usr/sbin/sshd -T -C user=root,host=localhost,addr=127.0.0.1 |
  grep -E '^(permitrootlogin|pubkeyauthentication|authorizedkeysfile|authorizedkeyscommand|usepam|passwordauthentication|authenticationmethods|strictmodes) '

# Service startup otherwise sends authentication failures to a syslog daemon
# that may not exist in the container. Emit them in Vast's container logs.
touch /etc/default/ssh
if ! grep -q '# geodistill-auth-log' /etc/default/ssh; then
  cat >> /etc/default/ssh <<'EOF'

# geodistill-auth-log
SSHD_OPTS="${SSHD_OPTS:-} -E /proc/1/fd/1 -o LogLevel=VERBOSE"
EOF
fi
service ssh restart
export PATH=/opt/train-venv/bin:$PATH
test "$(cat /opt/source.sha256)" = f469deac24b3c19fa8a3e288f3262b26294860c855052fa442dbc48ffdfc449e
AUTO_TRAIN=0 bash /app/docker/vast-start.sh
