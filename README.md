# SnapEnhance AI

<p align="center">
  <img src="static/images/snapenhance-og-minimal.png" alt="SnapEnhance AI" width="600">
</p>

> **Note:** Some features are currently under development and being fixed, including advanced function calling and real-time data integration.

SnapEnhance AI is a web application providing an interface to interact with AI models via the Together AI API.

### Features
- Chat interface with multiple AI models.
- Function calling for weather, time, product search, and Wikipedia search.
- Clean, responsive UI.

### Local Development
**Requirements**:
- Python 3.9+
- Together AI API key

**Setup**:
1. Clone the repository.
2. Create a `.env` file using `.env.example` and add API keys.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `python server.py`
5. Access at http://localhost:5001

### Deployment to Vercel
**Requirements**:
- Vercel account
- Together AI API key

**Steps (CLI)**:
1. Fork or clone the repository.
2. Install Vercel CLI: `npm install -g vercel`
3. Deploy: `vercel`
4. Add environment variables in Vercel dashboard (from `.env.example`).

**Steps (Dashboard)**:
1. Create a new Vercel project.
2. Connect GitHub repository.
3. Configure:
   - Framework: Other
   - Build command: None
   - Output directory: None
   - Install command: `pip install -r requirements.txt`
4. Add environment variables.
5. Deploy.

### Environment Variables
- `TOGETHER_API_KEY` (required): Together AI API key.
- `OPENWEATHERMAP_API_KEY` (optional): Weather data.
- `TIMEZONEDB_API_KEY` (optional): Time data.
- `OPENCAGE_API_KEY` (optional): Geocoding.

### License
MIT