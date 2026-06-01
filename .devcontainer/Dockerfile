FROM ghcr.io/games-on-whales/gstreamer:1.26.7

ARG HOME="/home/$UNAME"
ENV HOME=$HOME

# Setup user
RUN <<_SETUP_USER
#!/bin/bash
set -e

userdel -r ubuntu
groupadd --gid $PGID $UNAME
useradd --uid $PUID --gid $PGID $UNAME
mkdir -p $HOME
chown -R $PUID:$PGID $HOME

apt-get update -y
apt-get install -y sudo
echo $UNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$UNAME
chmod 0440 /etc/sudoers.d/$UNAME
_SETUP_USER

# Dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
               curl git ca-certificates build-essential pkg-config libssl-dev libicu76 \
               libwayland-dev libwayland-server0 libinput-dev libxkbcommon-dev libgbm-dev \
               librust-gstreamer-allocators-dev

ARG RUST_VERSION=1.96.0
ENV RUST_VERSION=$RUST_VERSION
ENV PATH="$HOME/.cargo/bin:${PATH}"

USER $UNAME

RUN <<_INSTALL_RUST
#!/bin/bash
set -e

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
rustup install $RUST_VERSION
rustup default $RUST_VERSION

cargo install cargo-c
_INSTALL_RUST
