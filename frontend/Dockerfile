# frontend/Dockerfile
FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

# Install pnpm 9.0.4
RUN npm install -g pnpm@9.0.4

# Copy package.json only
COPY package.json ./

# Install dependencies without frozen-lockfile
RUN pnpm install

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app

# Install pnpm in builder stage
RUN npm install -g pnpm@9.0.4

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js collects completely anonymous telemetry data about general usage.
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

# Disable ESLint during build
ENV DISABLE_ESLINT_PLUGIN=true

# Build with ignore lint errors
RUN pnpm build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Create necessary directories
RUN mkdir -p public .next/static

# Copy files
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Only copy public if it exists and has content
RUN if [ -d /app/public ] && [ "$(ls -A /app/public)" ]; then \
    COPY --from=builder /app/public/ ./public/; \
    fi

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]