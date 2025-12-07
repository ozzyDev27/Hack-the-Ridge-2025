/**
 * Sixth Sense App - Main Application Component
 * 
 * This is the root component of the Sixth Sense application, a smart vision system
 * that provides real-time video monitoring with AI-powered scene understanding and
 * fall detection capabilities for enhanced safety and accessibility.
 * 
 * Features:
 * - Real-time video feed monitoring
 * - AI-powered scene description
 * - Object search and location
 * - Fall detection alerts
 * - Text-to-speech accessibility
 */

import { VideoFeed } from './components/VideoFeed';

/**
 * App Component
 * 
 * The main application container that sets up the layout and renders the header
 * and video feed components. Uses a mobile-first design with a fixed width of 390px.
 * 
 * @returns {JSX.Element} The main application UI
 */
export default function App() {
  return (
    <div className="w-[390px] mx-auto" style={{backgroundColor: '#fff1ae'}}>
      {/* Application Header - Contains branding and status indicator */}
      <header className="bg-white" style={{borderBottom: '1px solid #ae8ca3'}}>
        <div className="px-6 py-5 flex items-center justify-between">
          {/* App Title */}
          <h1 className="text-2xl font-bold" style={{color: '#474b5d'}}>Sixth Sense</h1>
          
          {/* Status Indicator - Pulsing dot shows system is active */}
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full animate-pulse" style={{backgroundColor: '#ae8ca3'}}></div>
          </div>
        </div>
      </header>

      {/* Main Content Area - Video feed and controls */}
      <VideoFeed />
    </div>
  );
}