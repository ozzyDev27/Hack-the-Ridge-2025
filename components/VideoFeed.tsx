/**
 * VideoFeed Component
 * 
 * A comprehensive video monitoring interface that provides:
 * - Live video streaming from camera feed
 * - Real-time fall detection monitoring
 * - AI-powered scene description via text-to-speech
 * - Object search and location functionality
 * - Environmental status display
 * 
 * This component interfaces with the backend API to provide intelligent
 * monitoring and accessibility features for users.
 */

import { Camera } from 'lucide-react';
import { useState, useEffect } from 'react';

// Backend API base URL - update this to match your backend server
const API_BASE = 'http://10.42.0.196:8000';

/**
 * VideoFeed Component
 * 
 * Renders the main video feed interface with controls for scene understanding
 * and fall detection monitoring.
 * 
 * @returns {JSX.Element} The video feed UI with controls
 */
export function VideoFeed() {
  // State management
  const [currentTime, setCurrentTime] = useState(new Date());
  const [textInput, setTextInput] = useState('');
  const [describeResponse, setDescribeResponse] = useState('');
  const [searchResponse, setSearchResponse] = useState('');
  const [fallDetected, setFallDetected] = useState(false);

  /**
   * Effect: Update current time display every second
   * Provides real-time timestamp overlay on video feed
   */
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  /**
   * Effect: Monitor fall detection status
   * Polls the backend API every 500ms to check for fall detection events
   * Updates the UI to alert users of potential falls
   */
  useEffect(() => {
    const checkFall = async () => {
      try {
        const response = await fetch(`${API_BASE}/fall`);
        const data = await response.json();
        setFallDetected(data.fall_detected || data.detected || false);
      } catch (error) {
        console.error('Error checking fall status:', error);
      }
    };

    // Poll for fall detection every 500ms
    const fallTimer = setInterval(checkFall, 500);
    checkFall(); // Initial call

    return () => clearInterval(fallTimer);
  }, []);

  /**
   * Convert text to speech using Web Speech API
   * Provides audio feedback for accessibility
   * 
   * @param {string} text - The text to be spoken aloud
   */
  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  /**
   * Handle object search in the current video frame
   * Sends the search query to the AI backend and speaks the result
   * to help users locate objects in their environment
   */
  const handleSearch = async () => {
    try {
      const response = await fetch(`${API_BASE}/search?object=${encodeURIComponent(textInput)}`);
      const data = await response.json();
      const textToSpeak = data.caption || data.result || JSON.stringify(data);
      speakText(textToSpeak);
    } catch (error) {
      const errorMsg = 'Error fetching search results: ' + error;
      speakText(errorMsg);
    }
  };

  /**
   * Request AI description of the current scene
   * Generates a natural language description of what's visible in the video
   * and speaks it aloud for accessibility
   */
  const handleDescribe = async () => {
    try {
      const response = await fetch(`${API_BASE}/describe`);
      const data = await response.json();
      const textToSpeak = data.caption || data.result || JSON.stringify(data);
      speakText(textToSpeak);
    } catch (error) {
      const errorMsg = 'Error fetching description: ' + error;
      speakText(errorMsg);
    }
  };

  return (
    <div className="">
      {/* Video Stream Container */}
      <div className="relative aspect-[4/3]" style={{backgroundColor: '#f5e9a0'}}>
        {/* Embedded WebRTC video stream from camera */}
        <iframe
          src="http://10.42.0.1:1984/stream.html?src=linux_usbcam&mode=webrtc"
          className="w-full h-full"
          title="Video Feed"
          style={{border: 'none'}}
        />
        
        {/* Gradient overlays for visual polish */}
        <div className="absolute top-0 left-0 right-0 h-12 pointer-events-none" style={{background: 'linear-gradient(to bottom, white, transparent)'}} />
        <div className="absolute bottom-0 left-0 right-0 h-12 pointer-events-none" style={{background: 'linear-gradient(to top, #fff1ae, transparent)'}} />

        {/* Live timestamp overlay - shows current date and time */}
        <div className="absolute top-3 left-3 bg-white/95 px-3 py-1 rounded-full text-xs font-semibold shadow-sm" style={{color: '#474b5d'}}>
          {currentTime.toLocaleString()}
        </div>
      </div>

      {/* System Status Dashboard */}
      <div className="px-6 py-5">
        <div className="grid grid-cols-2 gap-3">
          {/* Fall Detection Status Card */}
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Fall Detected?</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>
              {fallDetected ? 'Yes' : 'No'}
            </div>
          </div>
          
          {/* System Status Card */}
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Status</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>Online</div>
          </div>
        </div>
      </div>

      {/* Control Panel - Search and Describe functionality */}
      <div className="px-6 pb-6">
        {/* Object Search Input and Button */}
        <div className="flex gap-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Search for something..."
            className="flex-1 bg-white rounded-full px-5 py-3 text-base focus:outline-none focus:ring-2 border-2 font-medium shadow-sm"
            style={{color: '#474b5d', borderColor: '#ae8ca3', '--tw-ring-color': '#ae8ca3'} as any}
          />
          <button
            onClick={handleSearch}
            className="text-white font-semibold px-6 py-3 rounded-full text-base transition-colors shadow-md hover:opacity-90"
            style={{backgroundColor: '#474b5d'}}
          >
            Search
          </button>
        </div>
        
        {/* Scene Description Button - Describes current view */}
        <button
          onClick={handleDescribe}
          className="w-full mt-3 text-white font-semibold py-3 rounded-full text-base transition-colors shadow-md hover:opacity-90"
          style={{backgroundColor: '#ae8ca3'}}
        >
          Describe
        </button>
      </div>
    </div>
  );
}
