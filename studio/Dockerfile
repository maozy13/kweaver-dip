FROM node:22-bookworm-slim AS base

ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME:$PATH

RUN corepack enable && corepack prepare pnpm@10.28.2 --activate

# ── Stage 1: production dependencies ────────────────────────────

FROM base AS prod-deps

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./

RUN pnpm config set registry https://registry.npmmirror.com \
    && pnpm install --frozen-lockfile --prod \
    && pnpm store prune

# ── Stage 2: builder ────────────────────────────────────────────

FROM base AS builder

WORKDIR /build

COPY mcporter.json package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.build.json tsconfig.json ./
COPY assets ./assets
COPY built-in ./built-in
COPY extensions ./extensions
COPY scripts ./scripts
COPY src ./src
COPY templates ./templates

# Bundled OpenClaw plugin: copy deploy/openclaw-extensions/dip into studio/extensions/dip
# before docker build (see README); CI does this automatically.
#RUN test -d extensions/dip \
#  || (echo "ERROR: extensions/dip is missing. From monorepo root: mkdir -p studio/extensions && cp -R deploy/openclaw-extensions/dip studio/extensions/dip" >&2; exit 1)

RUN pnpm config set registry https://registry.npmmirror.com \
    && pnpm install --frozen-lockfile \
    && pnpm run build

# ── Stage 3: runtime ────────────────────────────────────────────

FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production

WORKDIR /app

COPY --from=builder /build/package.json ./package.json
COPY --from=builder /build/mcporter.json /root/.mcporter/mcporter.json
COPY --from=builder /build/dist ./dist
COPY --from=builder /build/assets ./assets
COPY --from=builder /build/templates ./templates
COPY --from=builder /build/built-in ./built-in
COPY --from=builder /build/extensions ./extensions
COPY --from=builder /build/scripts ./scripts
COPY --from=prod-deps /app/node_modules ./node_modules

RUN ln -sf /app/node_modules/openclaw/openclaw.mjs /usr/local/bin/openclaw \
    && ln -sf /app/node_modules/@kweaver-ai/kweaver-sdk/bin/kweaver.js /usr/local/bin/kweaver \
    && ln -sf /app/node_modules/mcporter/dist/cli.js /usr/local/bin/mcporter \
    && chmod +x /app/scripts/docker-entrypoint.sh /app/scripts/openclaw-init.sh

EXPOSE 3000 3001

CMD ["/app/scripts/docker-entrypoint.sh"]
