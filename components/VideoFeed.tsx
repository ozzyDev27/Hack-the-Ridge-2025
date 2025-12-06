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
      <div className="relative aspect-[4/3] flex items-center justify-center" style={{backgroundColor: '#f5e9a0'}}>
        {/* Simulated room scene */}
        <div className="relative w-full h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-28 h-28 mx-auto bg-white rounded-full flex items-center justify-center shadow-md">
              <Camera className="w-14 h-14" style={{color: '#ae8ca3'}} />
            </div>
          </div>
        </div>

        {/* Timestamp overlay */}
        <div className="absolute top-3 left-3 bg-white/95 px-3 py-1 rounded-full text-xs font-semibold shadow-sm" style={{color: '#474b5d'}}>
          {currentTime.toLocaleString()}
        </div>
      </div>

      {/* Environmental Stats */}
      <div className="px-6 py-5">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Motion</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>Active</div>
          </div>
          <div className="bg-white rounded-2xl p-4 text-center shadow-sm">
            <div className="text-sm mb-1 font-medium" style={{color: '#474b5d'}}>Status</div>
            <div className="text-lg font-bold" style={{color: '#ae8ca3'}}>Safe</div>
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
            onClick={() => console.log('Sent:', textInput)}
            className="text-white font-semibold px-6 py-3 rounded-full text-base transition-colors shadow-md hover:opacity-90"
            style={{backgroundColor: '#474b5d'}}
          >
            Search
          </button>
        </div>
        <button
          onClick={() => console.log('Describe')}
          className="w-full mt-3 text-white font-semibold py-3 rounded-full text-base transition-colors shadow-md hover:opacity-90"
          style={{backgroundColor: '#ae8ca3'}}
        >
          Describe
        </button>
      </div>
    </div>
  );
}
