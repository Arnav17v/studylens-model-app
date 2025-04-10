# -*- coding: utf-8 -*-
#this
import cv2
import time
import sys
import logging
import math
import numpy as np
import dlib
from imutils import face_utils
from deepface import DeepFace
import os
import requests
import threading # Re-introducing threading properly
import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog
import traceback # For detailed error printing

# --- Helper function ---
def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- Configuration ---
DEEPFACE_MODEL_NAME = "VGG-Face"; DEEPFACE_DETECTOR_BACKEND = "opencv"
TARGET_EMOTION = ["neutral","fear"]
DLIB_SHAPE_PREDICTOR_PATH = resource_path("shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 0.25; EYE_AR_CONSEC_FRAMES = 5; DISTRACTION_CONSEC_FRAMES = 10
# --- IMPORTANT: REPLACE WITH YOUR LIVE URL ---
#WEBSITE_API_URL = "https://studylens-api.onrender.com/api/v1/sessions/" # <-- EXAMPLE! REPLACE!
# --- Or for local testing: ---
WEBSITE_API_URL = "https://studylens-backend.onrender.com/api/v1/sessions/"

# --- Global Variables ---
# Session Control
session_active = False
stop_requested = False
analysis_thread = None
session_username = None



# Data Storage
session_summary_data = None

# Shared Data & Synchronization
status_lock = threading.Lock()
current_status = {
    "session_elapsed_sec": 0.0, "focused_time_sec": 0.0, # Display label is "Focused"
    "wasted_time_sec": 0.0,     # Display label is "Wasted"
    "drowsy_time_sec": 0.0, "wasted_percentage": 0.0, # Note: percentage is based on the accumulated time now labeled "focused"
    "dominant_emotion": "N/A", "is_drowsy": False, "is_distracted": False,
    "ongoing_attention_span_sec": 0.0, "last_attention_span_sec": 0.0,
    "max_attention_span_sec": 0.0, "analysis_running": False,
    "error_message": None,
}

# GUI Elements
root = None; status_label = None
start_button, stop_button, send_button = None, None, None

# Tkinter StringVars
session_time_var=None; focused_time_var=None; wasted_time_var=None;
wasted_perc_var=None; emotion_var=None; drowsy_var=None;
distracted_var=None; drowsy_time_var_display=None;
last_span_var=None; max_span_var=None; avg_span_var=None;

# --- dlib Initialization ---
dlib_detector, dlib_predictor = None, None
lStart, lEnd, rStart, rEnd = None, None, None, None
def initialize_dlib():
    # ... (Initialization code remains the same) ...
    global dlib_detector, dlib_predictor, lStart, lEnd, rStart, rEnd
    dat_file_path = DLIB_SHAPE_PREDICTOR_PATH
    try:
        print(f"[Init] Loading dlib predictor: {dat_file_path}")
        if not os.path.exists(dat_file_path): raise FileNotFoundError(f"File not found: {dat_file_path}")
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor(dat_file_path)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        print("[Init] dlib loaded.")
        return True
    except Exception as e:
        error_message = f"Error loading dlib:\n{e}\nPath: {dat_file_path}"
        if root and root.winfo_exists(): messagebox.showerror("dlib Error", error_message)
        else: print(f"ERROR during init: {error_message}")
        return False

# --- Helper Functions ---
def euclidean_distance(pointA, pointB): return np.linalg.norm(pointA - pointB)
def calculate_ear_dlib(eye): A, B=euclidean_distance(eye[1], eye[5]), euclidean_distance(eye[2], eye[4]); C=euclidean_distance(eye[0], eye[3]); return (A+B)/(2.0*C) if C>0 else 1.0

# --- Analysis Function (Background Thread) ---
def run_analysis_logic():
    """Handles webcam, analysis, and updates shared status in a background thread."""
    global session_active, stop_requested, session_summary_data, current_status, status_lock

    print("[Analysis Thread] Started.")
    cap = None

    # --- Internal Tracking Variables ---
    thread_session_start_time = None; thread_last_frame_time = None
    # SWAPPED: Accumulate time matching the *original* wasting conditions into this variable
    thread_accumulated_time_for_focused_label = 0.0
    thread_total_drowsy_time = 0.0 # Drowsy time remains independent
    thread_ear_counter = 0; thread_no_face_counter = 0
    # Attention span logic still tracks periods of ORIGINAL focus
    thread_was_originally_focused_last = False
    thread_original_focus_start_time = None
    thread_last_span = 0.0; thread_max_span = 0.0
    thread_total_attention_span_duration = 0.0; thread_completed_attention_spans = 0
    last_deepface_time = 0

    # --- Status Update Functions (remain the same) ---
    def update_shared_status(key, value):
        with status_lock: current_status[key] = value
    def update_multiple_shared_status(data_dict):
        with status_lock: current_status.update(data_dict)

    update_shared_status("analysis_running", True); update_shared_status("error_message", None)

    try:
        print("[Analysis Thread] Initializing webcam...")
        cap = cv2.VideoCapture(0); time.sleep(0.5)
        if not cap.isOpened(): raise IOError("Could not open webcam.")
        print("[Analysis Thread] Webcam opened successfully.")

        while not stop_requested:
            current_monotonic_time = time.monotonic()

            if session_active:
                if thread_session_start_time is None: # Initialize on first frame
                    print("[Analysis Thread] First active frame. Initializing session state.")
                    thread_session_start_time = current_monotonic_time
                    thread_last_frame_time = thread_session_start_time
                    # Reset counters/timers
                    thread_accumulated_time_for_focused_label = 0.0; thread_total_drowsy_time = 0.0
                    thread_ear_counter = 0; thread_no_face_counter = 0
                    thread_was_originally_focused_last = False; thread_original_focus_start_time = None
                    thread_last_span = 0.0; thread_max_span = 0.0
                    thread_total_attention_span_duration = 0.0; thread_completed_attention_spans = 0
                    last_deepface_time = 0

                delta_time = current_monotonic_time - thread_last_frame_time
                thread_last_frame_time = current_monotonic_time
                session_elapsed_sec = current_monotonic_time - thread_session_start_time

                ret, frame = cap.read()
                if not ret: time.sleep(0.05); continue

                # --- Perform Analysis (Dlib, Deepface - logic remains the same) ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dlib_face_detected = False; is_drowsy = False
                rects = dlib_detector(gray, 0)
                if len(rects) > 0:
                    dlib_face_detected = True; thread_no_face_counter = 0; rect = rects[0]
                    shape = dlib_predictor(gray, rect); shape_np = face_utils.shape_to_np(shape)
                    leftEye, rightEye = shape_np[lStart:lEnd], shape_np[rStart:rEnd]
                    leftEAR, rightEAR = calculate_ear_dlib(leftEye), calculate_ear_dlib(rightEye)
                    average_ear = (leftEAR + rightEAR) / 2.0
                    if average_ear < EYE_AR_THRESH: thread_ear_counter += 1; is_drowsy = thread_ear_counter >= EYE_AR_CONSEC_FRAMES
                    else: thread_ear_counter = 0
                else: dlib_face_detected = False; thread_ear_counter = 0; thread_no_face_counter += 1

                is_target_emotion = False; dominant_emotion = "N/A"
                if dlib_face_detected:
                    try:
                        results = DeepFace.analyze(frame, ['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                        if results and results[0]: dominant_emotion = results[0].get('dominant_emotion', 'Error')
                        else: dominant_emotion = "No Face (DF)"
                        if dominant_emotion in TARGET_EMOTION: is_target_emotion = True
                    except ValueError as ve: dominant_emotion = "No Face (DF)" if "Face could not be detected" in str(ve) else f"DF Err: {ve}"
                    except Exception as df_err: dominant_emotion = f"DF Err: {df_err}"; print(f"[Analysis Thread] DeepFace Error: {df_err}")
                else: dominant_emotion = "No Face (dlib)"

                # --- Define ORIGINAL Wasting/Focusing Conditions ---
                is_distracted_condition = (thread_no_face_counter >= DISTRACTION_CONSEC_FRAMES)
                is_originally_wasting = not dlib_face_detected or \
                                        (dlib_face_detected and not is_target_emotion) or \
                                        is_drowsy or \
                                        is_distracted_condition
                is_originally_focused = not is_originally_wasting

                # --- Accumulate Times (SWAPPED LOGIC) ---
                # Accumulate time into 'focused label' variable when ORIGINAL wasting conditions are met
                if is_originally_wasting and delta_time > 0:
                    thread_accumulated_time_for_focused_label += delta_time
                # Drowsy time is still accumulated based on drowsiness condition
                if is_drowsy and delta_time > 0:
                    thread_total_drowsy_time += delta_time

                # --- Attention Span Calculation (Uses ORIGINAL focus conditions) ---
                ongoing_attention_span_sec = 0.0
                if is_originally_focused and not thread_was_originally_focused_last:
                    thread_original_focus_start_time = current_monotonic_time # Start span timer
                elif is_originally_wasting and thread_was_originally_focused_last:
                    if thread_original_focus_start_time is not None: # End span timer
                        current_span = current_monotonic_time - thread_original_focus_start_time
                        thread_last_span = round(current_span, 1)
                        thread_max_span = round(max(thread_max_span, current_span), 1)
                        thread_total_attention_span_duration += current_span
                        thread_completed_attention_spans += 1
                    thread_original_focus_start_time = None # Reset start time
                # Calculate ongoing span if currently in ORIGINAL focus state
                if is_originally_focused and thread_original_focus_start_time is not None:
                    ongoing_attention_span_sec = round(current_monotonic_time - thread_original_focus_start_time, 1)
                # Update state for next iteration
                thread_was_originally_focused_last = is_originally_focused

                # --- Calculate Display Values (SWAPPED) ---
                # Time displayed under "Focused Time" label is now the accumulated time from original 'wasting' states
                displayed_focused_time_sec = thread_accumulated_time_for_focused_label
                # Time displayed under "Wasted Time" label is the remainder
                displayed_wasted_time_sec = max(0, session_elapsed_sec - displayed_focused_time_sec)
                # Percentage displayed as "Wasted %" is based on the accumulated time (now labeled focused)
                displayed_wasted_perc = (displayed_focused_time_sec / session_elapsed_sec) * 100 if session_elapsed_sec > 0 else 0.0

                # --- Update Shared Status Dictionary (with SWAPPED values for focus/waste) ---
                live_stats = {
                    "session_elapsed_sec": round(session_elapsed_sec, 1),
                    "focused_time_sec": round(displayed_focused_time_sec, 1), # SWAPPED
                    "wasted_time_sec": round(displayed_wasted_time_sec, 1),   # SWAPPED
                    "drowsy_time_sec": round(thread_total_drowsy_time, 1),     # Unchanged
                    "wasted_percentage": round(displayed_wasted_perc, 1),      # Percentage based on new 'focused'
                    "dominant_emotion": dominant_emotion,                      # Unchanged trigger info
                    "is_drowsy": is_drowsy,                                    # Unchanged trigger info
                    "is_distracted": is_distracted_condition,                  # Unchanged trigger info
                    "ongoing_attention_span_sec": ongoing_attention_span_sec,  # Based on original focus
                    "last_attention_span_sec": thread_last_span,             # Based on original focus
                    "max_attention_span_sec": thread_max_span,             # Based on original focus
                    "error_message": None
                }
                update_multiple_shared_status(live_stats)

            else: # --- Session NOT Active ---
                if thread_session_start_time is not None: # Calculate final summary if session just ended
                    print("[Analysis Thread] Session stopped by GUI. Calculating final summary...")
                    end_time = current_monotonic_time

                    # Final attention span based on ORIGINAL focus state
                    if thread_was_originally_focused_last and thread_original_focus_start_time is not None:
                        final_span = end_time - thread_original_focus_start_time
                        thread_last_span = round(final_span, 1); thread_max_span = round(max(thread_max_span, final_span), 1)
                        thread_total_attention_span_duration += final_span; thread_completed_attention_spans += 1
                        print(f"[Analysis Thread] Final focus period end. Dur: {thread_last_span:.1f}s.")

                    # --- Final Summary Calculation (SWAPPED focus/waste) ---
                    total_elapsed = end_time - thread_session_start_time
                    # Final "Focused" value is the total accumulated time during original 'wasting' periods
                    final_focused_time = thread_accumulated_time_for_focused_label
                    # Final "Wasted" value is the remainder
                    final_wasted_time = max(0, total_elapsed - final_focused_time)
                    # Final percentage calculation based on the new 'focused' value
                    final_wasted_perc = (final_focused_time / total_elapsed) * 100 if total_elapsed > 0 else 0.0
                    # Average attention span still based on original focus periods
                    avg_attention_span_sec = round(thread_total_attention_span_duration / thread_completed_attention_spans, 1) if thread_completed_attention_spans > 0 else 0.0

                    # Store final summary data (with SWAPPED values for focus/waste)
                    session_summary_data = {
                        "total_duration_sec": round(total_elapsed, 2),
                        "wasted_time_sec":round(final_focused_time, 2),        # SWAPPED
                        "focused_time_sec":  round(final_wasted_time, 2),     # SWAPPED
                        "drowsy_time_sec": round(thread_total_drowsy_time, 2),   # Unchanged
                        "max_attention_span_sec": thread_max_span,           # Based on original focus
                        "avg_attention_span_sec": avg_attention_span_sec,    # Based on original focus
                        "wasted_percentage": round(final_wasted_perc, 1)       # Based on new 'focused'
                    }
                    print(f"[Analysis Thread] Final Session Data Calculated (Focus/Waste Swapped): {session_summary_data}")
                    thread_session_start_time = None # Signal summary calculated

                    if root and root.winfo_exists(): root.after(10, update_gui_after_stop) # Schedule final GUI update

                time.sleep(0.1) # Sleep longer when inactive

            time.sleep(0.01) # Small sleep even when active

    # --- Error Handling & Cleanup (remain the same) ---
    except Exception as e:
        print(f"[Analysis Thread] FATAL ERROR in analysis loop: {e}"); traceback.print_exc()
        update_shared_status("error_message", f"Analysis Error: {e}")
        if root and root.winfo_exists(): root.after(0, handle_analysis_error)
    finally:
        print("[Analysis Thread] Loop finished or error occurred.")
        if cap and cap.isOpened(): cap.release(); print("[Analysis Thread] Webcam released.")
        update_shared_status("analysis_running", False); print("[Analysis Thread] Finished.")


# --- GUI Update Function (Main Thread) ---
# This function remains the same - it displays whatever values are
# in current_status under the labels "Focused Time" and "Wasted Time".
def update_gui_stats():
    global session_active, current_status, status_lock, root
    if not session_active: return
    if root and root.winfo_exists():
        try:
            with status_lock: # Read current status safely
                elapsed = current_status["session_elapsed_sec"]
                focused = current_status["focused_time_sec"] # Reads the now-swapped value
                wasted = current_status["wasted_time_sec"]   # Reads the now-swapped value
                wasted_p = current_status["wasted_percentage"] # Reads the swapped percentage
                emotion = current_status["dominant_emotion"]
                drowsy = current_status["is_drowsy"]
                distracted = current_status["is_distracted"]
                drowsy_t = current_status["drowsy_time_sec"]
                last_span = current_status["last_attention_span_sec"] # Still original span
                ongoing_span = current_status["ongoing_attention_span_sec"] # Still original span
                max_span = current_status["max_attention_span_sec"] # Still original span
                err_msg = current_status["error_message"]

            if err_msg: status_label.config(text=f"Status: ERROR!"); return

            # Update StringVars - displays the swapped values under the original labels
            session_time_var.set(f"Session Time: {elapsed:.1f}s")
            focused_time_var.set(f"Wasted Time: {focused:.1f}s") # Shows accumulated original 'wasting' time
            wasted_time_var.set(f"Focused Time: {wasted:.1f}s")   # Shows remaining time
            wasted_perc_var.set(f"Wasted %: {wasted_p:.1f}%") # Shows percentage based on new 'focused'
            emotion_var.set(f"Emotion: {emotion}")
            drowsy_var.set(f"Drowsy: {'Yes' if drowsy else 'No'}")
            distracted_var.set(f"Distracted: {'Yes' if distracted else 'No'}")
            drowsy_time_var_display.set(f"Drowsy Time: {drowsy_t:.1f}s")
            ongoing_span_text = f" (Cur: {ongoing_span:.1f}s)" if ongoing_span > 0 else ""
            last_span_var.set(f"Last Span: {last_span:.1f}s{ongoing_span_text}") # Original span
            max_span_var.set(f"Max Span: {max_span:.1f}s") # Original span

            status_label.config(text="Status: Session Running...")
            if session_active: root.after(250, update_gui_stats)
        except Exception as e:
            print(f"[GUI Stats] Error updating GUI: {e}"); traceback.print_exc()
            if session_active and root.winfo_exists(): root.after(500, update_gui_stats)

# --- GUI Control Functions (Main Thread) ---
# (reset_globals_and_gui, update_gui_after_stop, handle_analysis_error, start_session, stop_session_logic, stop_session remain the same)
def reset_globals_and_gui():
    global session_summary_data, current_status, status_lock, session_username
    print("[GUI] Resetting state and labels.")
    session_summary_data = None
    session_username = None
    with status_lock:
        current_status = { # Reset to default values
            "session_elapsed_sec": 0.0, "focused_time_sec": 0.0, "wasted_time_sec": 0.0,
            "drowsy_time_sec": 0.0, "wasted_percentage": 0.0, "dominant_emotion": "N/A",
            "is_drowsy": False, "is_distracted": False, "ongoing_attention_span_sec": 0.0,
            "last_attention_span_sec": 0.0, "max_attention_span_sec": 0.0,
            "analysis_running": False, "error_message": None,
        }
    # Reset StringVars
    if session_time_var: session_time_var.set("Session Time: --")
    if focused_time_var: focused_time_var.set("Focused Time: --") # Label unchanged
    if wasted_time_var: wasted_time_var.set("Wasted Time: --")   # Label unchanged
    if wasted_perc_var: wasted_perc_var.set("Wasted %: --")
    if emotion_var: emotion_var.set("Emotion: N/A")
    if drowsy_var: drowsy_var.set("Drowsy: N/A")
    if distracted_var: distracted_var.set("Distracted: N/A")
    if drowsy_time_var_display: drowsy_time_var_display.set("Drowsy Time: --")
    if last_span_var: last_span_var.set("Last Span: --")
    if max_span_var: max_span_var.set("Max Span: --")
    if avg_span_var: avg_span_var.set("Avg Span: --")
    # Reset button/status states
    if status_label: status_label.config(text="Status: Ready")
    if start_button: start_button.config(state=tk.NORMAL)
    if stop_button: stop_button.config(state=tk.DISABLED)
    if send_button: send_button.config(state=tk.DISABLED)

def update_gui_after_stop():
    global session_summary_data
    print("[GUI Update] Updating GUI after stop confirmed by thread.")
    status_label.config(text="Status: Stopped. Data Ready.")
    if start_button: start_button.config(state=tk.NORMAL)
    if stop_button: stop_button.config(state=tk.DISABLED)
    if send_button: send_button.config(state=tk.NORMAL if session_summary_data else tk.DISABLED)
    if session_summary_data: # Display final span info (still based on original focus)
         max_s = session_summary_data.get('max_attention_span_sec', 0.0)
         avg_s = session_summary_data.get('avg_attention_span_sec', 0.0)
         if max_span_var: max_span_var.set(f"Max Span: {max_s:.1f}s")
         if avg_span_var: avg_span_var.set(f"Avg Span: {avg_s:.1f}s")
         if last_span_var: last_span_var.set("Last Span: --")
    else: # Clear span info if no summary
         if max_span_var: max_span_var.set("Max Span: --")
         if avg_span_var: avg_span_var.set("Avg Span: --")
         if last_span_var: last_span_var.set("Last Span: --")

def handle_analysis_error():
    print("[GUI] Handling analysis error.")
    with status_lock: err_msg = current_status.get("error_message", "Unknown analysis error.")
    status_label.config(text="Status: Analysis ERROR!")
    messagebox.showerror("Analysis Error", err_msg)
    stop_session_logic() # Ensure session stops


def start_session():
    global session_active, analysis_thread, stop_requested, session_username # Add session_username here
    if session_active:
        print("[GUI] Start clicked, but session already active.")
        return

    # --- Check if analysis thread is lingering ---
    if analysis_thread and analysis_thread.is_alive():
        print("[GUI] Warning: Analysis thread still alive? Waiting briefly...")
        analysis_thread.join(timeout=0.5)
        if analysis_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis thread seems stuck. Please wait or restart.")
            return

    # --- Reset state BEFORE getting new session info ---
    print("[GUI] Resetting previous session state before starting new one.")
    reset_globals_and_gui() # <-- MOVE RESET CALL HERE

    # --- Prompt for Username ---
    username_input = tkinter.simpledialog.askstring("Username Required",
                                                    "Please enter your username:",
                                                    parent=root)

    if username_input is None: # User pressed Cancel
        print("[GUI] Username input cancelled. Session not started.")
        # Reset button states since we aborted before starting
        if start_button: start_button.config(state=tk.NORMAL)
        if stop_button: stop_button.config(state=tk.DISABLED)
        if send_button: send_button.config(state=tk.DISABLED)
        if status_label: status_label.config(text="Status: Ready")
        return # Abort starting the session
    elif not username_input.strip(): # User entered nothing or only whitespace
        print("[GUI] Empty username entered. Using 'anonymous'.")
        session_username = "anonymous"
         # Optional: Show a warning if you prefer not to use anonymous automatically
         # messagebox.showwarning("Username", "No username entered. Proceeding as 'anonymous'.")
    else:
         session_username = username_input.strip() # Store the captured username globally
         print(f"[GUI] Username captured: {session_username}")
    # --- End Username Prompt ---


    # --- Proceed with Session Start ---
    print("[GUI] Start Session button clicked.");
    # Note: reset_globals_and_gui() was already called

    session_active = True;
    stop_requested = False
    status_label.config(text="Status: Starting Session...")
    start_button.config(state=tk.DISABLED);
    stop_button.config(state=tk.NORMAL);
    send_button.config(state=tk.DISABLED)

    print("[GUI] Starting analysis thread...")
    analysis_thread = threading.Thread(target=run_analysis_logic, daemon=True);
    analysis_thread.start()
    root.after(500, update_gui_stats) # Start GUI updates

def stop_session_logic():
    global session_active
    if not session_active: print("[GUI Logic] Stop called, but session not active."); return
    print("[GUI Logic] Signaling analysis thread to stop session...")
    session_active = False # Signal the analysis thread
    if status_label: status_label.config(text="Status: Stopping Session...")
    if stop_button: stop_button.config(state=tk.DISABLED)

def stop_session():
    print("[GUI] Stop Session button clicked."); stop_session_logic()

# --- Send Data Function (Remains the same) ---
def send_data_to_website():
    global session_summary_data, session_username # Add session_username here

    if not session_summary_data:
        messagebox.showwarning("No Data", "No session data to send.")
        return

    # --- Add check for username ---
    if not session_username:
        messagebox.showerror("Missing Info", "Username was not set for this session. Cannot send data.")
        print("[GUI] Send aborted: Username missing.")
        return
    # --- End check ---

    if not WEBSITE_API_URL or "YOUR_WEBSITE_API_ENDPOINT_HERE" in WEBSITE_API_URL :
         messagebox.showerror("Config Error", "API URL is not configured correctly.")
         return

    # --- Prepare data payload with username ---
    data_to_send = dict(session_summary_data) # Create a copy of the summary data
    data_to_send["username"] = session_username # Add the username field
    # --- End data preparation ---

    print(f"[GUI] Send button clicked. Starting send task for user '{session_username}' with data: {data_to_send}") # Log user and data
    status_label.config(text="Status: Sending data...");
    send_button.config(state=tk.DISABLED)

    # Pass the modified dictionary (with username) to the background task
    send_thread = threading.Thread(target=send_request_task, args=(data_to_send,), daemon=True);
    send_thread.start()

def send_request_task(data_to_send):
    global session_summary_data
    try:
        print(f"[Send Task] Sending data to {WEBSITE_API_URL}")
        headers = {'Content-Type': 'application/json'}
        response = requests.post(WEBSITE_API_URL, json=data_to_send, headers=headers, timeout=15); response.raise_for_status()
        print(f"[Send Task] Data sent successfully. Status: {response.status_code}")
        if root and root.winfo_exists():
            root.after(0, lambda: messagebox.showinfo("Success", f"Data sent successfully!\nServer status: {response.status_code}"))
            root.after(0, lambda: status_label.config(text="Status: Data Sent. Ready."))
    except requests.exceptions.RequestException as e:
        print(f"[Send Task] Error sending data: {e}"); error_message = f"Failed to send data:\n{e}"
        try:
            if e.response is not None: error_message = f"Failed (HTTP {e.response.status_code}):\n{e.response.json().get('detail', e.response.text)}"
        except Exception: pass
        if root and root.winfo_exists():
            root.after(0, lambda msg=error_message: messagebox.showerror("Send Error", msg))
            root.after(0, lambda: status_label.config(text="Status: Send Error"))
            root.after(0, lambda: send_button.config(state=tk.NORMAL) if session_summary_data else None)
    except Exception as e:
         print(f"[Send Task] Unexpected error sending data: {e}"); traceback.print_exc()
         if root and root.winfo_exists():
            root.after(0, lambda e=e: messagebox.showerror("Send Error", f"An unexpected error occurred:\n{e}"))
            root.after(0, lambda: status_label.config(text="Status: Send Error"))
            root.after(0, lambda: send_button.config(state=tk.NORMAL) if session_summary_data else None)

# --- Build GUI (Remains the same) ---
root = tk.Tk(); root.title("StudyLens Monitor (Threaded - Swapped F/W)"); root.geometry("400x370"); root.resizable(False, False)
logging.basicConfig(level=logging.INFO)
# StringVars
session_time_var=tk.StringVar(root); focused_time_var=tk.StringVar(root); wasted_time_var=tk.StringVar(root); wasted_perc_var=tk.StringVar(root); emotion_var=tk.StringVar(root); drowsy_var=tk.StringVar(root); distracted_var=tk.StringVar(root); drowsy_time_var_display=tk.StringVar(root); last_span_var=tk.StringVar(root); max_span_var=tk.StringVar(root); avg_span_var=tk.StringVar(root)
# Layout
status_label = tk.Label(root, text="Status: Initializing...", font=("Arial", 14, "bold")); status_label.pack(pady=(10, 5))
stats_frame = tk.Frame(root, padx=10, pady=5); stats_frame.pack(fill=tk.X, pady=5)
tk.Label(stats_frame, textvariable=session_time_var, font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=1); tk.Label(stats_frame, textvariable=emotion_var, font=("Arial", 10)).grid(row=0, column=1, sticky="w", padx=(20, 0), pady=1)
tk.Label(stats_frame, textvariable=focused_time_var, font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=1); tk.Label(stats_frame, textvariable=drowsy_var, font=("Arial", 10)).grid(row=1, column=1, sticky="w", padx=(20, 0), pady=1) # Label "Focused Time"
tk.Label(stats_frame, textvariable=wasted_time_var, font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=1); tk.Label(stats_frame, textvariable=distracted_var, font=("Arial", 10)).grid(row=2, column=1, sticky="w", padx=(20, 0), pady=1) # Label "Wasted Time"
tk.Label(stats_frame, textvariable=wasted_perc_var, font=("Arial", 10)).grid(row=3, column=0, sticky="w", pady=1); tk.Label(stats_frame, textvariable=drowsy_time_var_display, font=("Arial", 10)).grid(row=3, column=1, sticky="w", padx=(20, 0), pady=1) # Label "Wasted %"
tk.Label(stats_frame, textvariable=last_span_var, font=("Arial", 10)).grid(row=4, column=0, sticky="w", pady=1); tk.Label(stats_frame, textvariable=max_span_var, font=("Arial", 10)).grid(row=4, column=1, sticky="w", padx=(20, 0), pady=1)
tk.Label(stats_frame, textvariable=avg_span_var, font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5,1))
stats_frame.columnconfigure(0, weight=1); stats_frame.columnconfigure(1, weight=1)
button_frame = tk.Frame(root); button_frame.pack(pady=10)
start_button = tk.Button(button_frame, text="Start Session", command=start_session, width=12, state=tk.DISABLED); start_button.grid(row=0, column=0, padx=10)
stop_button = tk.Button(button_frame, text="Stop Session", command=stop_session, width=12, state=tk.DISABLED); stop_button.grid(row=0, column=1, padx=10)
send_button = tk.Button(root, text="Send Last Session Data", command=send_data_to_website, width=28, state=tk.DISABLED); send_button.pack(pady=10)

# --- Main Execution (Remains the same) ---
if __name__ == "__main__": # Corrected __name__ check
    print("Application starting...")
    def on_closing(): # (on_closing remains the same)
        global stop_requested, session_active, analysis_thread
        if messagebox.askokcancel("Quit", "Do you want to quit?\nUnsent session data will be lost."):
            print("Quit request received via window close.")
            stop_requested = True; session_active = False
            if analysis_thread and analysis_thread.is_alive():
                print("Waiting for analysis thread to finish..."); analysis_thread.join(timeout=1.5)
                if analysis_thread.is_alive(): print("Warning: Analysis thread did not finish cleanly.")
            else: print("Analysis thread was not running or already finished.")
            print("Destroying Tkinter window..."); root.destroy(); print("Application cleanup complete.")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    if initialize_dlib():
        reset_globals_and_gui(); print("Initialization complete. Starting GUI main loop...")
        root.mainloop()
    else:
        print("Exiting due to dlib initialization failure.")
        if root and root.winfo_exists(): status_label.config(text="Status: dlib Init Failed!")
        time.sleep(2);
        if root and root.winfo_exists(): root.destroy()
    print("Application finished.")