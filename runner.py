import subprocess
import re
import requests
import sys

# --- CONFIGURATION ---
CF_API_TOKEN = "your_api_token_here"
CF_ZONE_ID = "your_zone_id_here"
RECORD_NAME = "camera-app.snorlax.codes"
LOCAL_PORT = "5000" # The port your local full-stack app runs on

HEADERS = {
    "Authorization": f"Bearer {CF_API_TOKEN}",
    "Content-Type": "application/json"
}

def get_cf_record_id():
    """Queries Cloudflare to find the unique ID of the CNAME record."""
    url = f"https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/dns_records?name={RECORD_NAME}&type=CNAME"
    res = requests.get(url, headers=HEADERS).json()
    if res.get('success') and len(res.get('result', [])) > 0:
        return res['result'][0]['id']
    else:
        print(f"Error: Could not find CNAME record for {RECORD_NAME}.")
        sys.exit(1)

def update_cf_record(record_id, target_hostname):
    """Updates the CNAME target via Cloudflare API."""
    url = f"https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/dns_records/{record_id}"
    payload = {
        "type": "CNAME",
        "name": RECORD_NAME,
        "content": target_hostname,
        "ttl": 60,
        "proxied": False # Must be False for Quick Tunnels
    }
    res = requests.put(url, headers=HEADERS, json=payload).json()
    if res.get('success'):
        print(f"SUCCESS: {RECORD_NAME} now points to {target_hostname}")
    else:
        print(f"FAILED to update DNS record: {res.get('errors')}")

def run_tunnel_and_update_dns(record_id):
    """Executes cloudflared, scrapes the URL, and updates DNS."""
    # cloudflared prints its connection info to stderr
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{LOCAL_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Regex to capture the trycloudflare.com hostname
    url_pattern = re.compile(r"https://([a-zA-Z0-9-]+\.trycloudflare\.com)")
    
    print("Starting cloudflared and waiting for URL allocation...")
    
    # Read stderr line-by-line as the process runs
    for line in process.stderr:
        print(line.strip()) # Print tunnel logs to console
        match = url_pattern.search(line)
        if match:
            target_hostname = match.group(1)
            print(f"\n--- Intercepted URL: {target_hostname} ---")
            update_cf_record(record_id, target_hostname)
            break # Stop reading output once we have the URL; let the tunnel run
            
    # Keep the main thread alive while the subprocess runs
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nTerminating tunnel...")
        process.terminate()

if __name__ == "__main__":
    record_id = get_cf_record_id()
    run_tunnel_and_update_dns(record_id)