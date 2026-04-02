FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Runtime: nginx serves the built static files ──
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
# SPA fallback — all routes serve index.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
