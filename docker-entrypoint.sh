#!/bin/sh
# Fix ownership of the bind-mounted volumes: the host owns ./exodus-data and
# ./exodus-models with the host user's uid/gid, which may not match the uid of
# the image's `exodus` user.  Nobody expects root to own these (mlama-vulkan
# reads models, and identity.key has to be created), so chown then drop
# privileges to the unprivileged `exodus` user for the real command.

set -e

chown -R exodus:exodus /data /models

exec setpriv --reuid exodus --regid exodus --init-groups \
    /usr/local/bin/exodus "$@"