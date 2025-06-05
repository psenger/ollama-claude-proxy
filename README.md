# 🧠 Open WebUI + Pipelines (Docker Compose Setup)

This project configures and launches two related services using Docker Compose:

- `open-webui`: A UI-based interface for interacting with language models.
- `open-webui-pipeline`: A custom extension that processes data pipelines via LLM APIs.

The services are configured to persist data, stay up to date with the latest code, and restart automatically if they crash or your system restarts.

## 📦 Structure Overview

```
/Developer/open-webui/
├── docker-compose.yaml         # Main Docker Compose configuration
├── .env                        # Secure storage for secrets (e.g., API keys)
├── startup.sh                  # Script to update, rebuild, and run containers
├── stop.sh                     # Script to stop and remove containers
├── open-webui/                 # Contains Dockerfile and source for WebUI
├── open-webui-pipelines/       # Contains Dockerfile and source for Pipelines
```

## 🚀 Starting the Services

Use the `startup.sh` script to:

- Pull the latest code from Git.
- Rebuild Docker images.
- Launch or restart both containers.

```bash
./startup.sh
```

This ensures that you're always running the latest version of each service.

## 🛑 Stopping the Services

Use the `stop.sh` script to:

* Stop and remove the containers.
* Clean up any orphaned containers.

```bash
./stop.sh
```

This does **not** remove any volumes or persistent data. Your files in:

* `/Users/psenger/Documents/open-webui/data`
* `/Users/psenger/Documents/open-webui-pipeline`

...are safe and remain untouched.


## 🔐 Environment Variables

Secrets such as API keys are stored in a `.env` file located in the root project directory. **Do not commit this file to source control.**

Example `.env`:

```dotenv
ANTHROPIC_API_KEY=your_actual_key_here
```

This file is used automatically by Docker Compose.


## 🛠️ Additional Notes

* Ports exposed:

  * `open-webui` → [http://localhost:3333](http://localhost:3333)
  * `open-webui-pipeline` → [http://localhost:9099](http://localhost:9099)
* Containers are labelled with their original source and mount paths for easy tracking.
* Restart policies are set to `always`, so containers will restart on reboot or crash.

## ✅ Good Practices

* Run `./startup.sh` periodically to ensure your containers are up to date.
* Use `./stop.sh` before making manual Docker changes or cleaning up.
* Review logs with `docker logs open-webui` or `docker logs open-webui-pipeline` if needed.

## 🧽 Optional Cleanup (Advanced)

If you want to remove all unused Docker data (be **careful**, this is irreversible):

```bash
docker system prune -a
docker volume prune
```

## 📦 Future Improvements

You might consider:

* Adding Slack or email notifications on container lifecycle events.
* Setting up `systemd` to launch `startup.sh` on system boot.
* Logging actions from `startup.sh` and `stop.sh` to a file for audit purposes.

---

© 2025 — Maintained by \[psenger]



