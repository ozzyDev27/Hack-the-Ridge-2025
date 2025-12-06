import { VideoFeed } from './components/VideoFeed';

export default function App() {
  return (
    <div className="bg-amber-50 w-[428px] mx-auto">
      {/* Header */}
      <header className="bg-orange-50 border-b border-orange-200">
        <div className="px-6 py-5 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-orange-900">Sixth Sense</h1>
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-semibold text-blue-600">Live</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <VideoFeed />
    </div>
  );
}