import socket
import cv2
import numpy as np
import threading
import time
from queue import Queue

# Drone config
DRONE_IP = "192.168.1.1"
UDP_PORT = 8080
RTSP_PORT = 7070

# Frame queue for threading (non-blocking)
frame_queue = Queue(maxsize=2)  # Keep only latest frames
stats = {'fps': 0, 'frame_count': 0}

def send_stream_start():
    """Send initialization command to drone"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        init_cmd = bytes([0x03, 0x66, 0x00, 0x00, 0x00, 0x00, 0x66, 0x99])
        sock.sendto(init_cmd, (DRONE_IP, 7099))
        sock.close()
    except:
        pass

def rtsp_capture_thread():
    """Capture frames in separate thread for zero blocking"""
    rtsp_url = f"rtsp://{DRONE_IP}:{RTSP_PORT}/webcam"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # CRITICAL: Set minimal buffer for lowest latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Only 1 frame buffer = lowest latency
    
    fps_counter = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            fps_counter += 1
            stats['frame_count'] += 1
            
            # Calculate FPS
            if time.time() - fps_time >= 1.0:
                stats['fps'] = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # Drop old frames, keep only latest (critical for real-time)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Discard old frame
                except:
                    pass
            
            try:
                frame_queue.put_nowait(frame)
            except:
                pass
    
    cap.release()

def enhance_fast(frame):
    """Ultra-fast enhancement - only sharpening, no slow denoising"""
    # Simple unsharp mask - very fast
    gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
    sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    
    # Quick contrast boost
    return cv2.convertScaleAbs(sharpened, alpha=1.15, beta=5)

# Configure for fastest possible connection
print("=" * 60)
print("🚁 RC UFO - Ultra Low Latency Feed")
print("=" * 60)
print("Connecting...")

# Try RTSP first (usually most reliable for your drone)
rtsp_url = f"rtsp://{DRONE_IP}:{RTSP_PORT}/webcam"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Critical settings for speed
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # MINIMAL buffer = lowest latency
cap.set(cv2.CAP_PROP_FPS, 30)            # Request max FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # MJPEG = faster decode

if cap.isOpened():
    print("✓ Connected!")
    print("\nControls:")
    print("  [q] Quit")
    print("  [e] Toggle enhancement (ON by default)")
    print("  [s] Screenshot")
    print("  [f] Toggle FPS display")
    print()
    
    # Start capture in background thread
    cap.release()  # Close test connection
    capture_thread = threading.Thread(target=rtsp_capture_thread, daemon=True)
    capture_thread.start()
    
    enhancement_enabled = True
    show_fps = True
    last_frame = None
    
    try:
        while True:
            # Get latest frame (non-blocking)
            if not frame_queue.empty():
                frame = frame_queue.get()
                last_frame = frame
                
                # Fast enhancement if enabled
                if enhancement_enabled:
                    frame = enhance_fast(frame)
                
                # Lightweight overlay
                if show_fps:
                    cv2.putText(frame, f'{stats["fps"]:.0f} FPS', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Drone Feed', frame)
            elif last_frame is not None:
                # Show last frame if queue is empty
                cv2.imshow('Drone Feed', last_frame)
            
            # Fast key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                enhancement_enabled = not enhancement_enabled
                print(f"Enhancement: {'ON' if enhancement_enabled else 'OFF'}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS Display: {'ON' if show_fps else 'OFF'}")
            elif key == ord('s') and last_frame is not None:
                filename = f"drone_{int(time.time())}.jpg"
                cv2.imwrite(filename, last_frame)
                print(f"📸 Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped")
    
    print(f"\n📊 Total frames: {stats['frame_count']}")

else:
    # Fallback to UDP
    print("RTSP failed, trying UDP...")
    send_stream_start()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.1)  # Very short timeout for responsiveness
    
    print("✓ UDP listening...")
    print("Press 'q' to quit\n")
    
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    fps_counter += 1
                    if time.time() - fps_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_time = time.time()
                    
                    # Quick sharpen
                    frame = enhance_fast(frame)
                    
                    cv2.putText(frame, f'{current_fps} FPS', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Drone Feed (UDP)', frame)
            
            except socket.timeout:
                pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped")
    
    sock.close()

cv2.destroyAllWindows()
print("✓ Done!")
