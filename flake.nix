{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      nix-filter,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        gst-wayland-display = pkgs.rustPlatform.buildRustPackage {
          pname = "gst-wayland-display";
          version = "0.4.0";

          src = nix-filter {
            root = self;

            include = [
              (nix-filter.lib.inDirectory "c-bindings")
              (nix-filter.lib.inDirectory "gst-plugin-wayland-display")
              (nix-filter.lib.inDirectory "wayland-display-core")

              "Cargo.lock"
              "Cargo.toml"
            ];
          };

          cargoHash = "sha256-VjtrS0wmG9heZqb68GLM4PE6IezalQ3Z9CEYeJ/ndZw=";

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];

          buildInputs = with pkgs; [
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            libgbm
            libinput
            libxkbcommon
            udev
            wayland
          ];

          # Checks don't work properly in the Nix sandbox.
          doCheck = false;
        };
      in
      {
        formatter = pkgs.nixfmt-tree;

        devShells.default = pkgs.mkShell {
          inputsFrom = [
            gst-wayland-display
          ];

          packages = with pkgs; [
            gst_all_1.gst-plugins-good
          ];

          LD_LIBRARY_PATH = "${pkgs.libglvnd}/lib";
        };

        packages = {
          inherit gst-wayland-display;
          default = gst-wayland-display;
        };
      }
    );
}
