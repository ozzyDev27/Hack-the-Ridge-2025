import { Camera } from 'lucide-react';
import { useState, useEffect } from 'react';

export function VideoFeed() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [textInput, setTextInput] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="bg-gray-900">
      {/* Video Container */}
      <div className="relative bg-black aspect-video flex items-center justify-center">
        {/* Placeholder video feed */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900"></div>
        
        {/* Simulated room scene */}
        <div className="relative w-full h-full flex items-center justify-center">
          <div className="text-center space-y-6">
            <div className="w-40 h-40 mx-auto bg-gray-700 rounded-full flex items-center justify-center">
              <Camera className="w-20 h-20 text-gray-500" />
            </div>
          </div>
        </div>

        {/* Timestamp overlay */}
        <div className="absolute top-2 left-2 bg-black bg-opacity-30 text-white px-4 py-1 rounded-lg text-base font-mono">
          {currentTime.toLocaleString()}
        </div>

        {/* Status indicator */}
        <div className="absolute top-6 right-6">
        </div>
      </div>

      {/* Environmental Stats */}
      <div className="bg-gray-800 px-6 py-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 rounded-xl p-5 text-center">
            <div className="text-gray-400 text-base mb-2">Motion</div>
            <div className="text-green-500 text-xl font-semibold">Active</div>
          </div>
          <div className="bg-gray-900 rounded-xl p-5 text-center">
            <div className="text-gray-400 text-base mb-2">Status</div>
            <div className="text-white text-xl font-semibold">Safe</div>
          </div>
        </div>
      </div>

      {/* Text Input */}
      <div className="bg-gray-800 px-6 pb-6">
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Search for something..."
          className="w-full bg-gray-900 text-white rounded-xl p-5 text-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
        />
        <button
          onClick={() => console.log('Sent:', textInput)}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-xl text-xl transition-colors"
        >
          Search
        </button>
      </div>
    </div>
  );
}
