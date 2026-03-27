FROM golang:1.26-alpine AS builder

WORKDIR /app

# Copy go.mod and go.sum first (Docker cache layer)
# If these don't change, Docker skips re-downloading dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the code
COPY . .

# Build a single binary — this is Go's superpower
# One file, no dependencies, runs anywhere
RUN go build -o energy-forecaster .

# ── Second stage — tiny runtime image ──────────────────────
# The builder image is ~800MB. We only need the binary.
# This final image is ~20MB.
FROM alpine:latest

WORKDIR /app

# Copy only the compiled binary from the builder
COPY --from=builder /app/energy-forecaster .

EXPOSE 8080

CMD ["./energy-forecaster"]