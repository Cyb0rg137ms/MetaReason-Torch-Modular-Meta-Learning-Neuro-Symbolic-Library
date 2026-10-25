FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY metareason ./metareason
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[stats]"
ENTRYPOINT ["metareason"]
CMD ["bench", "all", "--quick"]
