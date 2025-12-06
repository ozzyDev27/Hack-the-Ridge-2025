import { VideoFeed } from './components/VideoFeed';
import { ActivityFeed } from './components/ActivityFeed';

export default function App() {
  return (
    <div className="min-h-screen bg-gray-900 w-[428px] mx-auto">
     

      {/* Header */}
      <header className="bg-gray-900 text-white px-6 py-6 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Sixth Sense</h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-lg font-medium text-gray-300">Live</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main>
        <VideoFeed />
        <ActivityFeed />
      </main>
    </div>
  );
}