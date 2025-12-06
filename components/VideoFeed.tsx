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
      <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 aspect-video flex items-center justify-center border-b border-slate-700/50">
        {/* Placeholder video feed */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-800 via-slate-900 to-slate-950"></div>
        
        {/* Simulated room scene */}
        <div className="relative w-full h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-blue-500/20 to-purple-600/20 rounded-2xl flex items-center justify-center border border-slate-600/30 backdrop-blur-sm">
              <Camera className="w-16 h-16 text-slate-400" />
            </div>
          </div>
        </div>

        {/* Timestamp overlay */}
        <div className="absolute top-3 left-3 bg-slate-950/60 backdrop-blur-md text-slate-200 px-3 py-1.5 rounded-lg text-sm font-medium border border-slate-700/50">
          {currentTime.toLocaleString()}
        </div>
      </div>

      {/* Environmental Stats */}
      <div className="px-4 py-5">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 text-center border border-slate-700/30">
            <div className="text-slate-400 text-sm font-medium mb-1">Motion</div>
            <div className="text-emerald-400 text-lg font-bold">Active</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 text-center border border-slate-700/30">
            <div className="text-slate-400 text-sm font-medium mb-1">Status</div>
            <div className="text-slate-200 text-lg font-bold">Safe</div>
          </div>
        </div>
      </div>

      {/* Text Input */}
      <div className="px-4 pb-5">
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Search for something..."
          className="w-full bg-slate-800/50 backdrop-blur-sm text-slate-200 rounded-lg p-4 text-base resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 border border-slate-700/30 placeholder:text-slate-500"
          rows={2}
        />
        <div className="grid grid-cols-2 gap-3 mt-3">
          <button
            onClick={() => console.log('Sent:', textInput)}
            className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white font-semibold py-3 rounded-lg text-base transition-all shadow-lg shadow-blue-500/20"
          >
            Search
          </button>
          <button
            onClick={() => console.log('Describe')}
            className="bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-700 hover:to-emerald-600 text-white font-semibold py-3 rounded-lg text-base transition-all shadow-lg shadow-emerald-500/20"
          >
            Describe
          </button>
        </div>
      </div>
    </div>
  );
}
