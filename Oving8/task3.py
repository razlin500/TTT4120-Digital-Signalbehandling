import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter
import pysptk

# Rasmus Nummelin X Claude

# Parameters
Fs = 8000  # Sampling frequency in Hz
AR_ORDER = 10  # AR model order for vowels

def load_vowel(filename):
    """Load a vowel sound from file"""
    data, fs = sf.read(filename)
    if fs != Fs:
        print(f"Warning: File sample rate {fs} Hz, expected {Fs} Hz")
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data[:, 0]
    return data

def record_vowel(duration=2.0):
    """Record a vowel from microphone"""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * Fs), samplerate=Fs, channels=1)
    sd.wait()
    print("Recording finished!")
    return recording.flatten()

def extract_ar_coefficients(signal):
    """Extract AR coefficients using Linear Predictive Coding"""
    # The pysptk.lpc function returns [b, a1, a2, ..., ap]
    # where b is the gain and a1...ap are the AR coefficients
    # Remember that a0 = 1
    coeffs = pysptk.sptk.lpc(signal.astype(np.float64), AR_ORDER)
    
    # coeffs[0] is the gain (b), coeffs[1:] are [a1, a2, ..., a10]
    # For the filter, we need [1, a1, a2, ..., a10]
    a = np.concatenate([[1], coeffs[1:]])
    gain = coeffs[0]
    
    return a, gain

def generate_excitation(signal_length, pitch_period=None):
    """Generate excitation signal (white noise or periodic impulses)"""
    # For simplicity, use white noise excitation
    # For better results, you could use pitch detection and periodic impulses
    excitation = np.random.randn(signal_length)
    return excitation

def transform_vowel(source_vowel, target_vowel):
    """Transform source vowel to sound like target vowel"""
    # Extract AR coefficients from target vowel (spectral envelope)
    a_target, gain_target = extract_ar_coefficients(target_vowel)
    
    # Extract AR coefficients from source vowel
    a_source, gain_source = extract_ar_coefficients(source_vowel)
    
    # Method 1: Inverse filter source, then filter with target
    # Step 1: Apply inverse filter of source to get excitation
    excitation = lfilter(a_source, [1], source_vowel)
    
    # Step 2: Filter excitation with target coefficients
    transformed = lfilter([1], a_target, excitation)
    
    # Normalize to prevent clipping
    transformed = transformed * 0.9 / np.max(np.abs(transformed))
    
    return transformed

def play_sound(signal, fs=Fs):
    """Play audio signal"""
    sd.play(signal, fs)
    sd.wait()

# Example usage
if __name__ == "__main__":
    print("=== Vowel Transformer ===")
    print("\nThis program transforms vowels using AR modeling.")
    
    # Example 1: Transform between two pre-recorded vowels
    print("\n--- Example 1: Transform pre-recorded vowels ---")
    print("Load your vowel files (e.g., 'vowel_a.wav', 'vowel_i.wav')")
    
    # Uncomment and modify these lines with your actual file paths:
    # source_file = 'vowel_a.wav'  # The vowel you want to transform FROM
    target_file = 'Oving8/a.wav'  # The vowel you want to transform TO
    # 
    # source_vowel = load_vowel(source_file)
    # target_vowel = load_vowel(target_file)
    # 
    # print(f"\nPlaying original source vowel...")
    # play_sound(source_vowel)
    # 
    # print(f"Transforming vowel...")
    # transformed_vowel = transform_vowel(source_vowel, target_vowel)
    # 
    # print(f"Playing transformed vowel...")
    # play_sound(transformed_vowel)
    
    # Example 2: Record your own vowel and transform it
    print("\n--- Example 2: Record and transform your own vowel ---")
    print("Instructions:")
    print("1. First, you'll record a vowel in your voice")
    print("2. Then, load a target vowel file to transform to")
    
    # Uncomment to use:
    input("Press Enter to start recording your vowel (say 'aaaa' for 2 seconds)...")
    my_vowel = record_vowel(duration=2.0)
    # 
    print("Playing back your recording...")
    play_sound(my_vowel)
    # 
    # # Load target vowel
    # 
    # target_file = 'target_vowel_e.wav'  # Change to your target vowel
    target_vowel = load_vowel(target_file)
    # 
    print("Transforming your vowel...")
    my_transformed = transform_vowel(my_vowel, target_vowel)
    # 
    print("Playing transformed version...")
    play_sound(my_transformed)
    
    print("\n=== Setup Instructions ===")
    print("1. Download vowel samples from It's Learning")
    print("2. Uncomment the example code sections above")
    print("3. Replace file paths with your actual vowel files")
    print("4. Run the script with headphones and microphone ready")
    print("\nNote: The AR[10] model captures the spectral envelope (formants)")
    print("which characterizes different vowels.")