import { VideoFeed } from './components/VideoFeed';

export default function App() {
  return (
    <div className="w-[428px] mx-auto" style={{backgroundColor: '#fff1ae'}}>
      {/* Header */}
      <header className="bg-white" style={{borderBottom: '1px solid #ae8ca3'}}>
        <div className="px-6 py-5 flex items-center justify-between">
          <h1 className="text-2xl font-bold" style={{color: '#474b5d'}}>Sixth Sense</h1>
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full animate-pulse" style={{backgroundColor: '#ae8ca3'}}></div>
            <span className="text-sm font-semibold" style={{color: '#ae8ca3'}}>Live</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <VideoFeed />
    </div>
  );
}