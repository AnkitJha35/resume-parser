FROM node:22-alpine

WORKDIR /app

# Install dependencies first for efficient layer caching
COPY package*.json ./
RUN npm install

# Copy application source
COPY . .

EXPOSE 5173

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:5173/ >/dev/null || exit 1

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
