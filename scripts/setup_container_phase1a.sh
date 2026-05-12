#!/usr/bin/env bash
# Phase 1a container setup — installs PHP scoring toolchain + Python ML deps
# inside the running unsloth-studio container.
#
# Usage (from host):
#   docker exec unsloth-studio bash /workspace/project/scripts/setup_container_phase1a.sh
#
# Idempotent: skips already-installed components. Safe to re-run after container
# restart (the bind-mount carries this script in via the launcher's
# `-v $PWD:/workspace/project` mount added in dgx-toolbox commit 6a673db).
#
# Resolves wp-finetune host-vs-container split for Phase 1a per
# DGX_TOOLBOX_ISSUES.md #10/#11 workaround narrative: container shell is the
# canonical scoring environment, studio UI is not needed.
set -euo pipefail

PHPCS_STANDARDS=(
    "wp-coding-standards/wpcs:^3"
    "automattic/vipwpcs:^3"
    "phpcompatibility/php-compatibility:^9"
    "pheromone/phpcs-security-audit:^2"
)

echo "[1/5] PHP runtime"
if command -v php >/dev/null 2>&1; then
    echo "  already installed: $(php --version | head -1)"
else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        php php-xml php-mbstring php-curl php-zip unzip >/dev/null
    echo "  installed: $(php --version | head -1)"
fi

echo "[2/5] composer"
if command -v composer >/dev/null 2>&1; then
    echo "  already installed: $(composer --version | head -1)"
else
    EXPECTED=$(php -r 'echo file_get_contents("https://composer.github.io/installer.sig");')
    php -r "copy('https://getcomposer.org/installer', '/tmp/composer-setup.php');"
    ACTUAL=$(php -r "echo hash_file('sha384', '/tmp/composer-setup.php');")
    if [ "$EXPECTED" != "$ACTUAL" ]; then
        echo "  ERROR: composer installer signature mismatch" >&2
        rm -f /tmp/composer-setup.php
        exit 1
    fi
    php /tmp/composer-setup.php --quiet --install-dir=/usr/local/bin --filename=composer
    rm /tmp/composer-setup.php
    echo "  installed: $(composer --version | head -1)"
fi

export PATH="$HOME/.composer/vendor/bin:$PATH"

echo "[3/5] PHPCS + WordPress / VIP / Security standards"
# Allow the composer plugin used by phpcs standards to auto-register installed_paths.
composer global config --no-plugins --no-interaction \
    allow-plugins.dealerdirect/phpcodesniffer-composer-installer true >/dev/null 2>&1 || true
if command -v phpcs >/dev/null 2>&1; then
    echo "  phpcs already installed: $(phpcs --version)"
else
    composer global require --no-progress --no-interaction \
        "squizlabs/php_codesniffer=^3.7" >/dev/null
    echo "  installed: $(phpcs --version)"
fi
INSTALLED_PATHS=""
for pkg in "${PHPCS_STANDARDS[@]}"; do
    name="${pkg%%:*}"
    spec="${pkg#*:}"
    dir="$HOME/.composer/vendor/${name}"
    if [ -d "$dir" ]; then
        echo "  standard $name already present"
    else
        composer global require --no-progress --no-interaction "${name}=${spec}" >/dev/null
        echo "  installed standard: $name"
    fi
    INSTALLED_PATHS="$INSTALLED_PATHS,$dir"
done
INSTALLED_PATHS="${INSTALLED_PATHS#,}"
phpcs --config-set installed_paths "$INSTALLED_PATHS" >/dev/null
echo "  phpcs registered standards: $(phpcs -i)"

echo "[4/5] PHPStan"
if command -v phpstan >/dev/null 2>&1; then
    echo "  already installed: $(phpstan --version)"
else
    composer global require --no-progress --no-interaction "phpstan/phpstan=^1" >/dev/null
    echo "  installed: $(phpstan --version)"
fi

echo "[5/5] Python ML deps for Phase 1a"
python3 -m pip install --quiet --no-deps \
    "xgboost>=2.1,<3" \
    "scikit-learn>=1.5,<2" \
    "pyyaml>=6"
python3 -c 'import xgboost, sklearn, yaml; print(f"  xgb {xgboost.__version__} | sklearn {sklearn.__version__} | yaml {yaml.__version__}")'

if ! grep -q ".composer/vendor/bin" /root/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.composer/vendor/bin:$PATH"' >> /root/.bashrc
fi

echo
echo "Phase 1a container setup complete. Open a shell with:"
echo "  docker exec -it unsloth-studio bash"
echo "Then cd /workspace/project and run the Phase 1a pipeline:"
echo "  RUBRIC_USE_LLM_CHECKS=1 python -m scripts.extract_pass_anchors --emit-features --output output/diagnostic/pass_anchors_features.jsonl --target-anchors 500 --sample-pool 1500"
echo "  RUBRIC_USE_LLM_CHECKS=1 python -m scripts.phase0_score_seeds --emit-features"
echo "  python -m scripts.build_calibration_dataset"
echo "  python -m scripts.calibrate_rubric"
