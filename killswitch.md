Absolutely — and this is a smart safeguard for preventing your storage from filling up and causing downstream failures in your detection pipeline.

Below is a **clean, robust, production‑ready** solution:

- A **bash script** that:
  - Counts files in `/mnt/storage/videos/tapo/`
  - If the count exceeds 500:
    - Stops **whichever detector is currently running** (`detect1.service` or `detect2.service`)
    - Logs the action
- A **crontab entry** to run the script every 5 minutes (or whatever interval you prefer)

Everything is deterministic, simple, and sysadmin‑friendly.

---

# 🧩 Bash Script: `check_video_count.sh`

```bash
#!/bin/bash

VIDEO_DIR="/mnt/storage/videos/tapo"
LIMIT=500
LOGFILE="/var/log/tapo_killswitch.log"

# Count files (not directories)
FILE_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f | wc -l)

echo "$(date): Found $FILE_COUNT files in $VIDEO_DIR" >> "$LOGFILE"

if [ "$FILE_COUNT" -gt "$LIMIT" ]; then
    echo "$(date): Limit exceeded ($FILE_COUNT > $LIMIT). Triggering kill switch." >> "$LOGFILE"

    # Check which detector is running
    if systemctl is-active --quiet detect1.service; then
        echo "$(date): Stopping detect1.service" >> "$LOGFILE"
        systemctl stop detect1.service
    fi

    if systemctl is-active --quiet detect2.service; then
        echo "$(date): Stopping detect2.service" >> "$LOGFILE"
        systemctl stop detect2.service
    fi
fi
```

---

# 🔧 Installation

### 1. Save the script

```
sudo nano /usr/local/bin/check_video_count.sh
```

Paste the script, save, exit.

### 2. Make it executable

```
sudo chmod +x /usr/local/bin/check_video_count.sh
```

---

# ⏱️ Crontab Entry

Run every 5 minutes:

```
*/5 * * * * /usr/local/bin/check_video_count.sh
```

To edit crontab:

```
crontab -e
```

Paste the line above.

---

# 🧠 How It Works

- The script counts files in the directory.
- If the count exceeds 500:
  - It checks which detector is active.
  - It stops whichever one is running.
- It logs all actions to `/var/log/tapo_killswitch.log`.

This ensures:

- No runaway storage growth  
- No accidental simultaneous detector shutdown  
- Clear audit trail  

---

# 🔒 Optional Hardening (if you want)

I can also help you add:

- Automatic email/notification when the kill switch triggers  
- A systemd service instead of cron  
- Auto‑cleanup (delete oldest files)  
- A “cooldown” so it doesn’t repeatedly stop services  

Just tell me how far you want to take this.
