import { Activity, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { useState, useEffect } from 'react';
import activitiesData from '../recentactivity.json';

interface ActivityItem {
  id: string;
  type: 'normal' | 'warning' | 'info';
  message: string;
  time: string;
}

export function ActivityFeed() {
  const [activities, setActivities] = useState<ActivityItem[]>([]);

  useEffect(() => {
    setActivities(activitiesData as ActivityItem[]);
  }, []);

  const getIcon = (type: ActivityItem['type']) => {
    switch (type) {
      case 'warning':
        return <AlertTriangle className="w-7 h-7 text-yellow-500" />;
      case 'info':
        return <Info className="w-7 h-7 text-blue-500" />;
      default:
        return <CheckCircle className="w-7 h-7 text-green-500" />;
    }
  };

  return (
    <div className="bg-gray-800 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Activity className="w-7 h-7 text-white" />
        <h2 className="text-white text-2xl font-bold">Recent Activity</h2>
      </div>
      <div className="space-y-4">
        {activities.map((activity) => (
          <div
            key={activity.id}
            className="bg-gray-900 rounded-xl p-5 flex items-start gap-4"
          >
            <div className="mt-1">{getIcon(activity.type)}</div>
            <div className="flex-1">
              <p className="text-white text-lg leading-relaxed">{activity.message}</p>
              <p className="text-gray-400 text-base mt-2">{activity.time}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
