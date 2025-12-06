import { VideoFeed } from './components/VideoFeed';

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 w-[428px] mx-auto">
      {/* Header */}
      <header className="bg-slate-950/50 backdrop-blur-sm border-b border-slate-700/50">
        <div className="px-5 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">SS</span>
            </div>
            <h1 className="text-xl font-semibold text-white tracking-tight">Sixth Sense</h1>
          </div>
          <div className="flex items-center gap-2 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-emerald-400">LIVE</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <VideoFeed />
    </div>
  );
}