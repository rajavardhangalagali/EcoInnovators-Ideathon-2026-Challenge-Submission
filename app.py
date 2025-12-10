"""
Solar Panel Detection System - Complete Web Application
Run this file to start the full system with web interface.
Everything is integrated here - just run: python app.py
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline components
from main_pipeline import SolarAnalysisPipeline, create_sample_input_file
from detection import ModelTrainer

# Import satellite fetcher for auto-fetch feature
try:
    from satellite_fetcher import create_satellite_fetcher
    SATELLITE_FETCHER_AVAILABLE = True
except ImportError:
    SATELLITE_FETCHER_AVAILABLE = False

app = Flask(__name__)

# Initialize pipeline (lazy loading)
pipeline = None

def get_pipeline():
    """Get or initialize the pipeline."""
    global pipeline
    if pipeline is None:
        try:
            pipeline = SolarAnalysisPipeline(config_path="configs/config.yaml")
        except Exception as e:
            print(f"Warning: Could not initialize pipeline: {e}")
            return None
    return pipeline

def reload_pipeline():
    """Force reload pipeline with latest model weights."""
    global pipeline
    pipeline = None
    return get_pipeline()

# ============================================================================
# HTML TEMPLATES
# ============================================================================

HOME_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar Panel Detection System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .section {
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        label {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        input[type="text"], input[type="file"], input[type="number"] {
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type=file] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            background: white;
        }
        .file-input-wrapper input[type=file]:hover {
            border-color: #667eea;
        }
        .file-input-label {
            display: none;
        }
        .result-box {
            margin-top: 20px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        .error-box {
            margin-top: 20px;
            padding: 20px;
            background: #ffebee;
            border-radius: 8px;
            border-left: 4px solid #f44336;
            color: #c62828;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .info-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .info-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        .status-verifiable {
            background: #4caf50;
            color: white;
        }
        .status-not-verifiable {
            background: #f44336;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 Solar Panel Detection System</h1>
        <p class="subtitle">AI-Powered Rooftop Solar Panel Analysis</p>

        <!-- Single Site Analysis -->
        <div class="section">
            <h2>📸 Analyze Single Site</h2>
            <form action="/analyze-single" method="post" enctype="multipart/form-data">
                <label>Sample ID:</label>
                <input type="text" name="sample_id" placeholder="e.g., 1001" required>
                
                <label>Latitude:</label>
                <input type="number" step="any" name="lat" placeholder="e.g., 37.4419" required>
                
                <label>Longitude:</label>
                <input type="number" step="any" name="lon" placeholder="e.g., -122.1430" required>
                
                <label>Upload Image (or leave empty to auto-fetch from satellite):</label>
                <input type="file" name="image" accept="image/*,.jpg,.jpeg,.png,.tif,.tiff" style="width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; cursor: pointer;">
                <p style="color: #666; font-size: 0.9em; margin-top: 5px;">🛰️ If no image uploaded, high-resolution satellite image will be auto-fetched via ESRI (100% FREE - no signup, no API key needed)</p>
                
                <button type="submit">🔍 Analyze Site</button>
            </form>
        </div>

        <!-- Batch Analysis -->
        <div class="section">
            <h2>📊 Batch Analysis</h2>
            <form action="/analyze-batch" method="post" enctype="multipart/form-data">
                <label>Upload CSV File (sample_id, lat, lon):</label>
                <input type="file" name="file" accept=".csv" required style="width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; cursor: pointer;">
                
                <label>Image Directory Path:</label>
                <input type="text" name="image_dir" placeholder="e.g., data/images/" value="data/input/" required>
                
                <button type="submit">🚀 Run Batch Analysis</button>
            </form>
        </div>

        <!-- System Info -->
        <div class="section">
            <h2>ℹ️ System Information</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3>System Status</h3>
                    <p>Pipeline: Ready</p>
                    <p>Model: YOLOv8</p>
                </div>
                <div class="info-card">
                    <h3>Quick Actions</h3>
                    <form action="/validate-config" method="post" style="margin-top: 10px;">
                        <button type="submit" style="width: 100%; padding: 10px;">Validate Config</button>
                    </form>
                </div>
                <div class="info-card">
                    <h3>Create Sample</h3>
                    <form action="/create-sample" method="post" style="margin-top: 10px;">
                        <button type="submit" style="width: 100%; padding: 10px;">Create Sample CSV</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with all forms."""
    return render_template_string(HOME_HTML)

@app.route('/analyze-single', methods=['POST'])
def analyze_single():
    """Analyze a single site for solar panels."""
    try:
        # Get form data
        sample_id = request.form.get('sample_id')
        lat = float(request.form.get('lat'))
        lon = float(request.form.get('lon'))
        verify_satellite = False
        
        # Handle image - either uploaded or auto-fetched from ESRI satellite
        image = request.files.get('image')
        upload_dir = Path('data/input/uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        if image and image.filename:
            # User uploaded an image
            image_path = upload_dir / f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image.save(str(image_path))
            image_source = "USER_UPLOAD"
        else:
            # Auto-fetch high-resolution satellite image from ESRI (completely free, no key needed)
            if not SATELLITE_FETCHER_AVAILABLE:
                return jsonify({'error': 'Satellite fetcher not available. Please upload an image.'}), 400
            
            try:
                # ESRI requires no API key - completely free public service
                fetcher = create_satellite_fetcher('esri')
                # High resolution: zoom 19, 1024x1024 for detailed rooftop view
                img_array, metadata = fetcher.fetch_image(lat=lat, lon=lon, zoom=19, width=1024, height=1024)
                
                # Save fetched image
                from PIL import Image
                image_path = upload_dir / f"{sample_id}_esri_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                Image.fromarray(img_array).save(str(image_path), 'JPEG', quality=95)
                image_source = "ESRI_SATELLITE"
                print(f"Auto-fetched high-res satellite image from ESRI for ({lat}, {lon})")
            except Exception as e:
                return jsonify({'error': f'Failed to fetch satellite image: {str(e)}. Please upload an image manually.'}), 400
        
        # Get pipeline
        pipe = get_pipeline()
        if not pipe:
            return jsonify({'error': 'Pipeline not initialized. Check configuration.'}), 500
        
        # Analyze site
        result = pipe.analyze_single_site(
            sample_id=sample_id,
            lat=lat,
            lon=lon,
            image_path=str(image_path),
            verify_with_satellite=verify_satellite
        )
        
        # Add image source to result
        if 'image_metadata' in result:
            result['image_metadata']['source'] = image_source
        
        # Save single site result to JSON file
        output_dir = Path('outputs/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file = output_dir / f"{sample_id}_result.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            print(f"Single site result saved to: {json_file}")
        except Exception as e:
            print(f"Warning: Could not save JSON result: {e}")
        
        # Check for verification errors
        if result.get('error') and 'verification' in result.get('error', '').lower():
            return render_result_page(result, is_error=True)
        
        # Return results page
        return render_result_page(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple sites from CSV file."""
    try:
        # Get uploaded CSV
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No CSV file uploaded'}), 400
        
        image_dir = request.form.get('image_dir', 'data/input/')
        
        # Save CSV
        csv_path = Path('data/input/uploaded_batch.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(csv_path))
        
        # Get pipeline
        pipe = get_pipeline()
        if not pipe:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        # Run batch analysis
        output_dir = 'outputs/results'
        results = pipe.analyze_batch(
            input_file=str(csv_path),
            image_dir=image_dir,
            output_dir=output_dir
        )
        
        # Ensure batch results file is saved
        batch_file = Path(output_dir) / 'batch_results.json'
        if not batch_file.exists() or batch_file.stat().st_size == 0:
            # Save results if file is missing or empty
            with open(batch_file, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
        
        # Return batch results
        if batch_file.exists() and batch_file.stat().st_size > 0:
            return send_file(str(batch_file), mimetype='application/json', as_attachment=True, download_name='batch_results.json')
        
        # Fallback: return JSON response
        return jsonify({
            'message': 'Batch analysis complete',
            'total_sites': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/validate-config', methods=['POST'])
def validate_config():
    """Validate system configuration."""
    try:
        pipe = get_pipeline()
        if not pipe:
            return jsonify({'error': 'Could not initialize pipeline'}), 500
        
        info = pipe.get_pipeline_info()
        return jsonify({
            'status': 'valid',
            'message': 'Configuration is valid',
            'info': info
        })
    except Exception as e:
        return jsonify({
            'status': 'invalid',
            'error': str(e)
        }), 500

@app.route('/create-sample', methods=['POST'])
def create_sample():
    """Create sample input CSV file."""
    try:
        output_path = 'data/input/sample_sites.csv'
        create_sample_input_file(output_path)
        return jsonify({
            'message': f'Sample file created at {output_path}',
            'path': output_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """REST API endpoint for analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sample_id = data.get('sample_id')
        lat = data.get('lat')
        lon = data.get('lon')
        image_path = data.get('image_path')
        # Production Feature: Satellite Verification (Currently Disabled)
        # verify_satellite = data.get('verify_satellite', False)
        verify_satellite = False
        
        if not all([sample_id, lat, lon, image_path]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        pipe = get_pipeline()
        if not pipe:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        result = pipe.analyze_single_site(
            sample_id=sample_id,
            lat=lat,
            lon=lon,
            image_path=image_path,
            verify_with_satellite=verify_satellite
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_result_page(result, is_error=False):
    """Render results page HTML."""
    if is_error:
        status_class = "status-not-verifiable"
        status_text = "VERIFICATION FAILED"
        bg_color = "#ffebee"
        border_color = "#f44336"
    else:
        status_class = "status-verifiable" if result.get('qc_status') == 'VERIFIABLE' else "status-not-verifiable"
        status_text = result.get('qc_status', 'UNKNOWN')
        bg_color = "#e8f5e9" if result.get('has_solar') else "#fff3e0"
        border_color = "#4caf50" if result.get('has_solar') else "#ff9800"
    
    # Build content sections separately to avoid nested f-string issues
    error_msg = ''
    if is_error:
        error_msg = f'<p style="color: #f44336; font-weight: 600; margin-top: 15px;">{result.get("error", "")}</p>'
    
    detection_section = ''
    if not is_error:
        has_solar_text = 'Yes' if result.get('has_solar') else 'No'
        confidence = round(result.get('confidence', 0), 3)
        panel_count = result.get('panel_count_est', 0)
        area = round(result.get('pv_area_sqm_est', 0), 1)
        capacity = round(result.get('capacity_kw_est', 0), 1)
        qc_status = result.get('qc_status', 'UNKNOWN')
        qc_notes = ', '.join(result.get('qc_notes', []))
        
        detection_section = f'''
                <div style="margin-top: 20px;">
                    <h3>Detection Results</h3>
                    <p><strong>Solar Panels Detected:</strong> {has_solar_text}</p>
                    <p><strong>Confidence:</strong> {confidence}</p>
                    <p><strong>Panel Count:</strong> {panel_count}</p>
                    <p><strong>Area:</strong> {area} m&sup2;</p>
                    <p><strong>Capacity:</strong> {capacity} kW</p>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>Quality Control</h3>
                    <p><strong>Status:</strong> {qc_status}</p>
                    <p><strong>Notes:</strong> {qc_notes}</p>
                </div>
                '''
    
    verification_section = ''
    if result.get('image_metadata', {}).get('image_verification'):
        verification_data = json.dumps(result.get('image_metadata', {}).get('image_verification', {}), indent=2)
        verification_section = f'''
                <div style="margin-top: 20px;">
                    <h3>Image Verification</h3>
                    <pre style="background: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
{verification_data}
                    </pre>
                </div>
                '''
    
    sample_id = result.get('sample_id', 'N/A')
    lat = result.get('lat', 'N/A')
    lon = result.get('lon', 'N/A')
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 40px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 30px;
            }}
            .result-box {{
                padding: 25px;
                background: {bg_color};
                border-radius: 10px;
                border-left: 4px solid {border_color};
                margin: 20px 0;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .info-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .status-badge {{
                display: inline-block;
                padding: 8px 20px;
                border-radius: 20px;
                font-weight: 600;
                margin: 10px 0;
            }}
            .status-verifiable {{
                background: #4caf50;
                color: white;
            }}
            .status-not-verifiable {{
                background: #f44336;
                color: white;
            }}
            .btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Analysis Results</h1>
            
            <div class="result-box">
                <h2>Sample ID: {sample_id}</h2>
                <p><strong>Coordinates:</strong> ({lat}, {lon})</p>
                
                <div class="status-badge {status_class}">
                    {status_text}
                </div>
                
                {error_msg}
                {detection_section}
                {verification_section}
            </div>
            
            <a href="/" class="btn">← Back to Home</a>
        </div>
    </body>
    </html>
    '''
    return html

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🔋 Solar Panel Detection System")
    print("=" * 60)
    print("\nStarting web server...")
    print("Visit: http://localhost:5000")
    print("\nPress CTRL+C to stop\n")
    
    # Create necessary directories
    Path('data/input/uploads').mkdir(parents=True, exist_ok=True)
    Path('outputs/results').mkdir(parents=True, exist_ok=True)
    Path('outputs/overlays').mkdir(parents=True, exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
