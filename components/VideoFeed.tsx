import { Camera } from 'lucide-react';
import { useState, useEffect } from 'react';

const API_BASE = 'http://10.42.0.196:8000';

export function VideoFeed() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [textInput, setTextInput] = useState('');
  const [describeResponse, setDescribeResponse] = useState('');
  const [searchResponse, setSearchResponse] = useState('');
  const [fallDetected, setFallDetected] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

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

    const fallTimer = setInterval(checkFall, 500);
    checkFall(); // Initial call

    return () => clearInterval(fallTimer);
  }, []);

  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

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
      {/* Video Container */}
      <div className="relative aspect-[4/3]" style={{backgroundColor: '#f5e9a0'}}>
        <iframe
          src="http://10.42.0.1:1984/stream.html?src=linux_usbcam&mode=webrtc"
          className="w-full h-full"
          title="Video Feed"
          style={{border: 'none'}}
        />
        
        {/* Gradient overlays */}
        <div className="absolute top-0 left-0 right-0 h-12 pointer-events-none" style={{background: 'linear-gradient(to bottom, white, transparent)'}} />
        <div className="absolute bottom-0 left-0 right-0 h-12 pointer-events-none" style={{background: 'linear-gradient(to top, #fff1ae, transparent)'}} />

        {/* Timestamp overlay */}
        <div className="absolute top-3 left-3 bg-white/95 px-3 py-1 rounded-full text-xs font-semibold shadow-sm" style={{color: '#474b5d'}}>
          {currentTime.toLocaleString()}
        </div>
      </div>

      {/* Environmental Stats */}
      <div className="px-6 py-5">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Fall Detected?</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>{fallDetected ? 'Yes' : 'No'}</div>
          </div>
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Status</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>Online</div>
          </div>
        </div>
      </div>

      {/* Text Input */}
      <div className="px-6 pb-6">
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
