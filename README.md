# SnapEnhance AI

SnapEnhance AI is a web application that provides an interface to interact with various AI models through the Together AI API.

## Features

- Chat interface with AI models
- Function calling capabilities for weather, time, product search, and Wikipedia search
- Clean and responsive UI
- Support for multiple AI models

## Local Development

### Prerequisites

- Python 3.9+
- Together AI API key

### Setup

1. Clone the repository
2. Create a `.env` file based on `.env.example` and add your API keys
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
python server.py
```

5. Access the application at http://localhost:5001

## Deployment to Vercel

### Prerequisites

- Vercel account
- Together AI API key

### Steps

1. Fork or clone this repository
2. Install Vercel CLI (optional):

```bash
npm install -g vercel
```

3. Deploy to Vercel:

```bash
vercel
```

4. Add environment variables in the Vercel dashboard:
   - Go to your project settings
   - Navigate to the "Environment Variables" tab
   - Add the required variables from `.env.example`

Alternatively, you can deploy directly from the Vercel dashboard:

1. Create a new project in Vercel
2. Connect your GitHub repository
3. Configure the project:
   - Framework preset: Other
   - Build command: None
   - Output directory: None
   - Install command: pip install -r requirements.txt
4. Add environment variables
5. Deploy

## Environment Variables

- `TOGETHER_API_KEY` (required): Your Together AI API key
- `OPENWEATHERMAP_API_KEY` (optional): For real weather data
- `TIMEZONEDB_API_KEY` (optional): For accurate time data
- `OPENCAGE_API_KEY` (optional): For geocoding

## License

MIT
