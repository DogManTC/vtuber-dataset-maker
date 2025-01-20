from flask import Flask, render_template_string, jsonify, request
import subprocess
import json

app = Flask(__name__)

# Initialize variables to track progress
progress_data = {
    "vod_id": "Loading...",
    "completed_vods": 0,
    "total_vods": 5,
    "vod_duration": "N/A",
    "total_audio_left": "N/A",
    "status": "Waiting..."
}

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loading Screen</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
        }

        .progress-container {
            position: relative;
            width: 300px;
            height: 300px;
        }

        .progress-container .circle {
            stroke-dasharray: 880;
            stroke-dashoffset: 880;
            stroke-width: 15;
            fill: none;
            stroke: gray;
        }

        .progress-container .lens {
            stroke-dasharray: 880;
            stroke-dashoffset: 880;
            stroke-width: 15;
            fill: none;
            stroke: url(#gradient);
        }

        .progress-container .background {
            stroke-width: 15;
            stroke: #444;
            fill: none;
        }

        .progress-container .percentage {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2em;
            font-weight: bold;
        }

        .info-text {
            margin-top: 20px;
            font-size: 1.2em;
            text-align: center;
        }

        .twitch-embed {
            margin-top: 20px;
            width: 300px;
            height: 169px; /* Aspect ratio 16:9 */
        }
    </style>
</head>
<body>
    <div id="loading-screen">
        <div class="progress-container">
            <svg width="300" height="300">
                <defs>
                    <linearGradient id="gradient" gradientTransform="rotate(90)">
                        <stop offset="5%" stop-color="red" />
                        <stop offset="20%" stop-color="orange" />
                        <stop offset="40%" stop-color="yellow" />
                        <stop offset="60%" stop-color="green" />
                        <stop offset="80%" stop-color="blue" />
                        <stop offset="95%" stop-color="indigo" />
                    </linearGradient>
                </defs>
                <circle class="background" cx="150" cy="150" r="140"></circle>
                <circle class="circle" cx="150" cy="150" r="140"></circle>
                <circle class="lens" cx="150" cy="150" r="140"></circle>
            </svg>
            <div class="percentage" id="percentage">0%</div>
        </div>
        <div class="info-text" id="info-text">Loading...</div>
        <div class="info-text" id="status-text">Status: Waiting...</div>
        <div id="twitch-embed" class="twitch-embed">
            <iframe
                src=""
                height="100%"
                width="100%"
                frameborder="0"
                allowfullscreen>
            </iframe>
        </div>
    </div>
    <script>
        const lens = document.querySelector('.lens');
        const percentageText = document.getElementById('percentage');
        const infoText = document.getElementById('info-text');
        const statusText = document.getElementById('status-text');
        const twitchEmbed = document.getElementById('twitch-embed').querySelector('iframe');

        // Store the last received data
        let lastData = null;

        function updateUI(data) {
            // Only update the UI if the data has changed
            if (JSON.stringify(data) === JSON.stringify(lastData)) {
                return;
            }
            lastData = data; // Update the lastData with the current data

            const { vod_id, completed_vods, total_vods, vod_duration, total_audio_left, status } = data;

            // Calculate progress percentage
            const progress = Math.min(100, Math.round((completed_vods / total_vods) * 100));

            // Update progress circle
            const offset = 880 - (880 * progress) / 100;
            lens.style.strokeDashoffset = offset;

            // Update percentage text
            percentageText.textContent = `${progress}%`;

            // Update Twitch embed
            twitchEmbed.src = `https://player.twitch.tv/?video=${vod_id}&parent=localhost`;

            // Update info text
            infoText.textContent = `
                VOD ID: ${vod_id}
                Completed VODs: ${Math.floor(completed_vods)}/${total_vods}
                Current VOD Duration: ${vod_duration}
                Total Audio Left: ${total_audio_left}
            `;

            // Update status text
            statusText.textContent = `Status: ${status}`;
        }

        function fetchProgress() {
            fetch('/progress')
                .then(response => response.json())
                .then(data => updateUI(data))
                .catch(error => console.error('Error fetching progress:', error));
        }

        setInterval(fetchProgress, 1000); // Update every second
    </script>
</body>
</html>
    ''')

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify(progress_data)

@app.route('/update_progress', methods=['POST'])
def update_progress():
    """
    Update the progress data from a POST request.
    The request body should contain:
    - vod_id
    - completed_vods
    - total_vods
    - vod_duration
    - total_audio_left
    - status
    """
    data = request.json
    if data:
        progress_data.update(data)
        return jsonify({"message": "Progress updated successfully!"}), 200
    return jsonify({"error": "Invalid data"}), 400

def update_website_with_progress(vod_id, status):
    """
    Updates the website with transcription progress via a JSON payload.

    Args:
        vod_id (str): The ID of the VOD being processed.
        status (str): The current status of the process.
    """
    try:
        vod_index = next((i for i, vod in enumerate(all_vods) if vod['url'].split("/videos/")[1] == vod_id), None)
        if vod_index is None:
            print(f"VOD ID {vod_id} not found in the list of VODs.")
            return

        vod = all_vods[vod_index]
        vod_num_in_list = vod_index + 1 + progress_data.get("stage_fraction", 0)
        vods_left = len(all_vods) - vod_num_in_list
        vod_duration_seconds = vod['duration_seconds']
        total_audio_left_seconds = sum(v['duration_seconds'] for v in all_vods[int(vod_num_in_list):])

        update_payload = {
            "vod_id": vod_id,
            "completed_vods": min(vod_num_in_list, len(all_vods)),
            "total_vods": len(all_vods),
            "vod_duration": format_duration(vod_duration_seconds),
            "total_audio_left": format_duration(total_audio_left_seconds),
            "status": status
        }

        print(f"Sending progress update to the website for VOD {vod_id} with status '{status}'...")
        curl_command = [
            "curl", "-X", "POST", "http://localhost:5000/update_progress",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(update_payload)
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully updated website with progress for VOD {vod_id}.")
        else:
            print(f"Failed to update website for VOD {vod_id}: {result.stderr}")
    except Exception as e:
        print(f"An error occurred while updating the website for VOD {vod_id}: {e}")

if __name__ == '__main__':
    app.run(debug=True)
