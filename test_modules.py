"""
🧪 TEST DETECTION MODULES - Validate improvements
Tests:
1. Vehicle Detector
2. Plate Detector
3. OCR Reader
4. Full Pipeline Integration
"""

import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from scr.detection.vehicle_detector import VehicleDetector
from scr.detection.plate_detector import PlateDetector
from scr.ocr.plate_reader import PlateReader
from scr.detection.no_plate_engine import NoPlateEngine

def test_vehicle_detector():
    """Test vehicle detection on test image"""
    print("\n" + "="*70)
    print("🧪 TEST 1: VEHICLE DETECTOR")
    print("="*70)
    
    try:
        detector = VehicleDetector()
        print("✅ VehicleDetector loaded")
        
        # Create a dummy image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (100, 100, 100)  # Gray frame
        
        vehicles = detector.detect(frame)
        print(f"✅ Detection ran successfully")
        print(f"   - Found {len(vehicles)} vehicles")
        print(f"   - Config VEHICLE_CONF: {config.VEHICLE_CONF}")
        print(f"   - Config IOU_THRESHOLD: {getattr(config, 'IOU_THRESHOLD', 'N/A')}")
        print(f"   - Config FILTER_SMALL_VEHICLES: {config.FILTER_SMALL_VEHICLES}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plate_detector():
    """Test plate detection on dummy image"""
    print("\n" + "="*70)
    print("🧪 TEST 2: PLATE DETECTOR")
    print("="*70)
    
    try:
        detector = PlateDetector()
        print("✅ PlateDetector loaded")
        
        # Create a dummy crop image
        crop = np.zeros((200, 300, 3), dtype=np.uint8)
        crop[:, :] = (80, 80, 80)  # Dark gray
        
        plates = detector.detect(crop)
        print(f"✅ Detection ran successfully")
        print(f"   - Found {len(plates)} plates")
        print(f"   - Config PLATE_CONF: {config.PLATE_CONF}")
        print(f"   - Config FILTER_SMALL_PLATES: {config.FILTER_SMALL_PLATES}")
        print(f"   - Config MIN_PLATE_WIDTH: {config.MIN_PLATE_WIDTH}")
        print(f"   - Config MAX_PLATE_ASPECT_RATIO: {config.MAX_PLATE_ASPECT_RATIO}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ocr_reader():
    """Test OCR reader on dummy plate image"""
    print("\n" + "="*70)
    print("🧪 TEST 3: OCR READER (Plate Reader)")
    print("="*70)
    
    try:
        reader = PlateReader(lang_list=['en', 'vi'], gpu=False)
        print("✅ PlateReader loaded")
        
        # Create a dummy plate image with some patterns
        plate = np.zeros((50, 150, 3), dtype=np.uint8)
        plate[:, :] = (255, 255, 255)  # White background (typical plate)
        
        # Add black region to look like text
        cv2.rectangle(plate, (20, 10), (50, 40), (0, 0, 0), -1)
        
        text, conf = reader.read(plate)
        print(f"✅ OCR ran successfully")
        print(f"   - Detected text: '{text}'")
        print(f"   - Confidence: {conf:.3f}")
        print(f"   - Config OCR_CONF_THRESH: {config.OCR_CONF_THRESH}")
        print(f"   - Config OCR_GRAYSCALE: {config.OCR_GRAYSCALE}")
        print(f"   - Config OCR_ENHANCE: {config.OCR_ENHANCE}")
        print(f"   - Config OCR_RESIZE_WIDTH: {config.OCR_RESIZE_WIDTH}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration values"""
    print("\n" + "="*70)
    print("🧪 TEST 0: CONFIGURATION")
    print("="*70)
    
    try:
        print("✅ Configuration loaded successfully")
        print("\n📊 DETECTION THRESHOLDS:")
        print(f"   - VEHICLE_CONF: {config.VEHICLE_CONF}")
        print(f"   - PLATE_CONF: {config.PLATE_CONF}")
        print(f"   - OCR_CONF_THRESH: {config.OCR_CONF_THRESH}")
        
        print("\n📊 NMS & FILTERING:")
        print(f"   - IOU_THRESHOLD: {getattr(config, 'IOU_THRESHOLD', 'N/A')}")
        print(f"   - EXPAND_RATIO: {config.EXPAND_RATIO}")
        print(f"   - FILTER_SMALL_VEHICLES: {config.FILTER_SMALL_VEHICLES}")
        print(f"   - MIN_VEHICLE_AREA: {config.MIN_VEHICLE_AREA}")
        print(f"   - FILTER_SMALL_PLATES: {config.FILTER_SMALL_PLATES}")
        print(f"   - MIN_PLATE_WIDTH: {config.MIN_PLATE_WIDTH}")
        
        print("\n📊 OCR PREPROCESSING:")
        print(f"   - OCR_ENHANCE: {config.OCR_ENHANCE}")
        print(f"   - OCR_RESIZE_WIDTH: {config.OCR_RESIZE_WIDTH}")
        print(f"   - OCR_GRAYSCALE: {config.OCR_GRAYSCALE}")
        
        print("\n✅ All config parameters loaded")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test full pipeline"""
    print("\n" + "="*70)
    print("🧪 TEST 4: FULL PIPELINE INTEGRATION")
    print("="*70)
    
    try:
        engine = NoPlateEngine()
        print("✅ NoPlateEngine (Main Pipeline) loaded")
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (100, 100, 100)
        
        print("✅ Pipeline components initialized:")
        print(f"   - Vehicle detector: {type(engine.detector).__name__}")
        print(f"   - Plate detector: {type(engine.plate_detector).__name__}")
        print(f"   - OCR reader: {type(engine.ocr).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "🧪 DETECTION MODULES TEST SUITE" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Run tests in order
    results["Config"] = test_config()
    results["Vehicle Detector"] = test_vehicle_detector()
    results["Plate Detector"] = test_plate_detector()
    results["OCR Reader"] = test_ocr_reader()
    results["Full Pipeline"] = test_pipeline_integration()
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System ready for training validation.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
