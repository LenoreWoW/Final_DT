# ─────────────────────────────────────────────────────────
# Quantum Digital Twin Platform - Frontend Dockerfile
# Multi-stage build for a lean production Next.js image
# ─────────────────────────────────────────────────────────

# Stage 1: Install dependencies
FROM node:20-alpine AS deps

WORKDIR /app

# Copy dependency manifests
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

# Install dependencies based on available lock file
RUN \
    if [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm install --frozen-lockfile; \
    else npm install; \
    fi


# Stage 2: Build the application
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Set build-time environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_API_URL=http://backend:8000

RUN npm run build


# Stage 3: Production runtime
FROM node:20-alpine AS runner

WORKDIR /app

# Metadata
LABEL maintainer="Quantum Digital Twin Platform Team"
LABEL description="Next.js frontend for the Quantum Digital Twin Platform"

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create a non-root user
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Copy only the necessary build artifacts
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
