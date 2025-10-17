# Stage 1: Build the application
FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.* ./
COPY ping_exporter.go .
RUN go mod tidy
RUN go build -o /ping_exporter_test .

# Stage 2: Create the final image
FROM alpine:latest
# Cài đặt ping và các công cụ mạng
RUN apk add --no-cache iputils
COPY --from=builder /ping_exporter_test /ping_exporter_test
EXPOSE 9999
CMD ["/ping_exporter_test"]
