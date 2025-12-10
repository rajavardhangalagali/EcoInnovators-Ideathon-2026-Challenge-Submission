"""
Command line interface for the Solar Panel Detection System.
Provides easy-to-use commands for batch processing and single site analysis.
"""

import click
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from main_pipeline import SolarAnalysisPipeline, create_sample_input_file
from detection import ModelTrainer


@click.group()
@click.version_option(version='1.0.0', prog_name='Solar Panel Detection System')
def cli():
    """
    Solar Panel Detection System
    
    An AI-powered system that analyzes satellite images to detect rooftop solar panels,
    providing quantification, quality assessment, and explainable results.
    """
    pass


@cli.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Path to CSV/JSON file containing site data (sample_id, lat, lon)')
@click.option('--image-dir', '-d', type=click.Path(), default='data/input/satellite_images',
              help='Directory for satellite images (will auto-fetch if images not found)')
@click.option('--output-dir', '-o', default='outputs/results', 
              help='Output directory for results (default: outputs/results)')
@click.option('--config', '-c', default='configs/config.yaml',
              help='Configuration file path (default: configs/config.yaml)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze_batch(input_file, image_dir, output_dir, config, verbose):
    """Analyze multiple sites from CSV/JSON input file."""
    
    if verbose:
        click.echo(f"Input file: {input_file}")
        click.echo(f"Image directory: {image_dir}")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"Config file: {config}")
    
    # Create image directory if it doesn't exist
    from pathlib import Path
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize pipeline
        click.echo("Initializing Solar Panel Analysis Pipeline...")
        click.echo("Note: Satellite images will be auto-fetched from ESRI if not found in image directory")
        pipeline = SolarAnalysisPipeline(config_path=config)
        
        if verbose:
            info = pipeline.get_pipeline_info()
            click.echo(f"Model: {info['model_info']['model_type']}")
            click.echo(f"Confidence threshold: {info['model_info']['confidence_threshold']}")
        
        # Run batch analysis
        click.echo("Starting batch analysis...")
        results = pipeline.analyze_batch(input_file, image_dir, output_dir)
        
        # Summary statistics
        total_sites = len(results)
        solar_detected = sum(1 for r in results if r['has_solar'])
        verifiable = sum(1 for r in results if r['qc_status'] == 'VERIFIABLE')
        
        click.echo(f"\n{'='*50}")
        click.echo("BATCH ANALYSIS COMPLETE")
        click.echo(f"{'='*50}")
        click.echo(f"Total sites analyzed: {total_sites}")
        click.echo(f"Sites with solar panels: {solar_detected} ({solar_detected/total_sites*100:.1f}%)")
        click.echo(f"Verifiable results: {verifiable} ({verifiable/total_sites*100:.1f}%)")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--sample-id', '-s', required=True, help='Sample ID for the site')
@click.option('--lat', required=True, type=float, help='Latitude coordinate')
@click.option('--lon', required=True, type=float, help='Longitude coordinate')
@click.option('--image-path', '-i', type=click.Path(exists=True),
              help='Path to satellite image (if not using image-dir)')
@click.option('--image-dir', '-d', type=click.Path(exists=True),
              help='Directory to search for image by sample-id')
@click.option('--output-file', '-o', help='Output JSON file path')
@click.option('--config', '-c', default='configs/config.yaml',
              help='Configuration file path (default: configs/config.yaml)')
@click.option('--show-overlay', is_flag=True, help='Generate and show overlay image')
@click.option('--verify-satellite', is_flag=True, 
              help='Verify uploaded image matches satellite image at coordinates (production feature)')
def analyze_single(sample_id, lat, lon, image_path, image_dir, output_file, config, show_overlay, verify_satellite):
    """Analyze a single site for solar panels."""
    
    if not image_path and not image_dir:
        click.echo("Error: Either --image-path or --image-dir must be provided", err=True)
        sys.exit(1)
    
    try:
        # Initialize pipeline
        click.echo("Initializing pipeline...")
        pipeline = SolarAnalysisPipeline(config_path=config)
        
        # Analyze site
        click.echo(f"Analyzing site {sample_id} at ({lat}, {lon})...")
        if verify_satellite:
            click.echo("🔍 Verifying image matches satellite location...")
        result = pipeline.analyze_single_site(
            sample_id, lat, lon, 
            image_path=image_path, 
            image_dir=image_dir,
            verify_with_satellite=verify_satellite
        )
        
        # Check for verification errors
        if result.get('error') and 'verification' in result.get('error', '').lower():
            click.echo(f"❌ Image verification failed!", err=True)
            click.echo(f"   {result.get('error')}", err=True)
            if 'image_verification' in result.get('image_metadata', {}):
                verification = result['image_metadata']['image_verification']
                click.echo(f"   Similarity score: {verification.get('similarity_score', 0):.2f}")
                click.echo(f"   Required threshold: {verification.get('threshold', 0.7):.2f}")
            return
        
        # Display results
        click.echo(f"\n{'='*50}")
        click.echo("ANALYSIS RESULTS")
        click.echo(f"{'='*50}")
        click.echo(f"Sample ID: {result['sample_id']}")
        click.echo(f"Coordinates: ({result['lat']}, {result['lon']})")
        click.echo(f"Solar panels detected: {'Yes' if result['has_solar'] else 'No'}")
        click.echo(f"Confidence: {result['confidence']:.3f}")
        click.echo(f"Panel count estimate: {result['panel_count_est']}")
        click.echo(f"Area estimate: {result['pv_area_sqm_est']:.1f} m²")
        click.echo(f"Capacity estimate: {result['capacity_kw_est']:.1f} kW")
        click.echo(f"QC Status: {result['qc_status']}")
        click.echo(f"QC Notes: {', '.join(result['qc_notes'])}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\nResults saved to: {output_file}")
        
        # Show overlay location
        if result['bbox_or_mask']:
            click.echo(f"Overlay image saved to: {result['bbox_or_mask']}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output-path', '-o', default='data/input/sample_sites.csv',
              help='Output path for sample file (default: data/input/sample_sites.csv)')
def create_sample(output_path):
    """Create a sample input CSV file for testing."""
    try:
        create_sample_input_file(output_path)
        click.echo(f"Sample input file created at: {output_path}")
        click.echo("You can now run: solar-detect analyze-batch -i {} -d <image_dir>".format(output_path))
    except Exception as e:
        click.echo(f"Error creating sample file: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--train-dir', required=True, type=click.Path(exists=True),
              help='Directory containing training images and labels')
@click.option('--val-dir', required=True, type=click.Path(exists=True),
              help='Directory containing validation images and labels')
@click.option('--epochs', default=100, help='Number of training epochs (default: 100)')
@click.option('--batch-size', default=16, help='Training batch size (default: 16)')
@click.option('--image-size', default=640, help='Training image size (default: 640)')
@click.option('--output-dir', default='models', help='Output directory for trained model')
@click.option('--base-model', default='models/yolov8_solar.pt', 
              help='Base model to start from (default: models/yolov8_solar.pt, or yolov8n.pt for fresh training)')
def train_model(train_dir, val_dir, epochs, batch_size, image_size, output_dir, base_model):
    """Train YOLOv8 model on solar panel data."""
    
    try:
        click.echo("Preparing model training...")

        # Always use fixed output directory and weights path
        fixed_model_dir = Path("models/yolov8_solar/weights")
        fixed_model_dir.mkdir(parents=True, exist_ok=True)
        fixed_best_pt = fixed_model_dir / "best.pt"

        # Check if base model exists, fallback to yolov8n.pt
        if not Path(base_model).exists():
            click.echo(f"Warning: {base_model} not found. Using yolov8n.pt instead.")
            base_model = 'yolov8n.pt'
        else:
            click.echo(f"Starting from existing model: {base_model}")

        # Initialize trainer with specified base model
        trainer = ModelTrainer(base_model=base_model)

        # Create dataset YAML
        dataset_yaml = trainer.create_dataset_yaml(train_dir, val_dir)
        click.echo(f"Created dataset configuration: {dataset_yaml}")

        # Train model, always output to fixed path
        click.echo("Starting model training...")
        click.echo("This may take several hours depending on your hardware...")

        model_path = trainer.train(
            dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size
        )

        # After training, copy best.pt to fixed path if needed
        from shutil import copyfile
        if Path(model_path).resolve() != fixed_best_pt.resolve():
            try:
                copyfile(model_path, fixed_best_pt)
                click.echo(f"Model copied to: {fixed_best_pt}")
            except Exception as copy_err:
                click.echo(f"Warning: Could not copy model to fixed path: {copy_err}")
        else:
            click.echo(f"Model already at: {fixed_best_pt}")

        # Update config.yaml automatically
        config_path = Path("configs/config.yaml")
        import yaml
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            if "model" in config_data:
                config_data["model"]["yolo_model_path"] = str(fixed_best_pt)
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                click.echo(f"config.yaml updated to use: {fixed_best_pt}")
            else:
                click.echo("Warning: 'model' section not found in config.yaml")
        else:
            click.echo("Warning: config.yaml not found, could not update model path.")

        click.echo(f"\nTraining complete! Model saved to: {fixed_best_pt}")

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml',
              help='Configuration file path (default: configs/config.yaml)')
def validate_config(config):
    """Validate configuration file and show pipeline information."""
    
    try:
        # Try to initialize pipeline
        pipeline = SolarAnalysisPipeline(config_path=config)
        info = pipeline.get_pipeline_info()
        
        click.echo(f"Configuration file: {config}")
        click.echo("✓ Configuration is valid")
        click.echo(f"\nPipeline Information:")
        click.echo(f"Model type: {info['model_info']['model_type']}")
        click.echo(f"Model path: {info['model_info']['model_path']}")
        click.echo(f"Device: {info['model_info']['device']}")
        click.echo(f"Confidence threshold: {info['model_info']['confidence_threshold']}")
        click.echo(f"Quantification assumption: {info['quantification_assumptions']['watts_per_sqm']} Wp/m²")
        
        # Check if model file exists
        model_path = Path(info['model_info']['model_path'])
        if model_path.exists():
            click.echo("✓ Model file exists")
        else:
            click.echo("⚠ Model file not found - you may need to train a model first")
            click.echo("  Use: solar-detect train-model --help")
        
    except Exception as e:
        click.echo(f"✗ Configuration error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--train-csv', required=True, type=click.Path(exists=True),
              help='Path to train_rooftop_data.csv')
@click.option('--test-csv', required=True, type=click.Path(exists=True), 
              help='Path to test_rooftop_data.csv')
@click.option('--image-dir', required=True, type=click.Path(exists=True),
              help='Directory containing rooftop images')
@click.option('--auto-train', is_flag=True, 
              help='Automatically start training after data processing')
def process_training_data(train_csv, test_csv, image_dir, auto_train):
    """Process the training dataset (train_rooftop_data.csv + test_rooftop_data.csv)."""
    
    try:
        from training_data_integration import TrainingDataProcessor
        
        click.echo("Processing training data...")
        
        # Initialize processor
        processor = TrainingDataProcessor()
        
        # Process training data
        click.echo(f"Processing training CSV: {train_csv}")
        train_stats = processor.process_training_csv(train_csv, image_dir)
        
        # Process test data
        click.echo(f"Processing test CSV: {test_csv}")
        test_stats = processor.process_test_csv(test_csv, image_dir)
        
        # Validate dataset
        validation = processor.validate_dataset()
        
        # Display results
        click.echo(f"\n{'='*50}")
        click.echo("DATA PROCESSING COMPLETE")
        click.echo(f"{'='*50}")
        click.echo(f"Training samples: {train_stats['train_samples']}")
        click.echo(f"Validation samples: {train_stats['val_samples']}")
        click.echo(f"Test samples: {test_stats['processed_test_samples']}")
        click.echo(f"Positive samples: {train_stats['positive_samples']}")
        click.echo(f"Negative samples: {train_stats['negative_samples']}")
        click.echo(f"Dataset ready for training: {validation['ready_for_training']}")
        
        if not validation['ready_for_training']:
            click.echo("⚠ Dataset validation failed. Check the processing logs.")
            return
        
        click.echo("✓ Dataset successfully processed and ready for training!")
        
        # Auto-train if requested
        if auto_train:
            click.echo("\nStarting automatic model training...")
            
            # Import here to avoid circular imports
            from detection import ModelTrainer
            
            trainer = ModelTrainer()
            dataset_yaml = "data/training/dataset.yaml"
            
            click.echo("Training model (this may take several hours)...")
            model_path = trainer.train(dataset_yaml, epochs=100, batch=16)
            
            click.echo(f"\n🎉 Model training complete!")
            click.echo(f"Model saved to: {model_path}")
            click.echo(f"Update your config.yaml to use: {model_path}")
        else:
            click.echo(f"\nTo train the model, run:")
            click.echo(f"python cli.py train-model --train-dir data/training/images/train --val-dir data/training/images/val")
        
    except Exception as e:
        click.echo(f"Error processing training data: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and help."""
    
    click.echo("Solar Panel Detection System v1.0.0")
    click.echo("=" * 40)
    click.echo()
    click.echo("This system analyzes satellite images to detect rooftop solar panels.")
    click.echo("It provides:")
    click.echo("• Solar panel detection using YOLOv8")
    click.echo("• Panel counting and area estimation")
    click.echo("• Capacity calculation (kW)")
    click.echo("• Quality control and verification")
    click.echo("• Explainable visualizations")
    click.echo()
    click.echo("Quick Start:")
    click.echo("1. solar-detect create-sample")
    click.echo("2. Add your images to data/input/")
    click.echo("3. solar-detect analyze-batch -i data/input/sample_sites.csv -d data/input/")
    click.echo()
    click.echo("For training a custom model:")
    click.echo("1. Prepare training data in YOLO format")
    click.echo("2. solar-detect train-model --train-dir <path> --val-dir <path>")
    click.echo()
    click.echo("For more information: solar-detect --help")


if __name__ == '__main__':
    cli()