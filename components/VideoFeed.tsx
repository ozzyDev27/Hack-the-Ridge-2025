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
    <div className="">
      {/* Video Container */}
      <div className="relative bg-amber-100 aspect-video flex items-center justify-center">
        {/* Simulated room scene */}
        <div className="relative w-full h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-28 h-28 mx-auto bg-orange-50 rounded-full flex items-center justify-center shadow-md">
              <Camera className="w-14 h-14 text-indigo-600" />
            </div>
          </div>
        </div>

        {/* Timestamp overlay */}
        <div className="absolute top-3 left-3 bg-orange-50/95 text-orange-900 px-3 py-1 rounded-full text-xs font-semibold shadow-sm">
          {currentTime.toLocaleString()}
        </div>
      </div>

      {/* Environmental Stats */}
      <div className="px-6 py-5">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-orange-50 rounded-2xl p-4 text-center shadow-sm">
            <div className="text-orange-800 text-sm mb-1 font-medium">Motion</div>
            <div className="text-blue-600 text-lg font-bold">Active</div>
          </div>
          <div className="bg-orange-50 rounded-2xl p-4 text-center shadow-sm">
            <div className="text-orange-800 text-sm mb-1 font-medium">Status</div>
            <div className="text-blue-600 text-lg font-bold">Safe</div>
          </div>
        </div>
      </div>

      {/* Text Input */}
      <div className="px-6 pb-6">
        <input
          type="text"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Search for something..."
          className="w-full bg-orange-50 text-orange-900 rounded-full px-5 py-3 text-base focus:outline-none focus:ring-2 focus:ring-indigo-400 border-2 border-orange-200 placeholder:text-orange-700 font-medium shadow-sm"
        />
        <div className="grid grid-cols-2 gap-3 mt-4">
          <button
            onClick={() => console.log('Sent:', textInput)}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-full text-base transition-colors shadow-md"
          >
            Search
          </button>
          <button
            onClick={() => console.log('Describe')}
            className="bg-indigo-500 hover:bg-indigo-600 text-white font-semibold py-3 rounded-full text-base transition-colors shadow-md"
          >
            Describe
          </button>
        </div>
      </div>
    </div>
  );
}
