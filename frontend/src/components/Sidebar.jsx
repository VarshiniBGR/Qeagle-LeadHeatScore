import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Upload, 
  Users
} from 'lucide-react';

const Sidebar = () => {
  const location = useLocation();
  
  const menuItems = [
    { path: '/dashboard', icon: Home, label: 'Overview', description: 'Dashboard & Status' },
    { path: '/upload', icon: Upload, label: 'Upload Leads', description: 'CSV File Upload' },
    { path: '/leads', icon: Users, label: 'Lead Results', description: 'Scored Leads' },
  ];
  
  const isActive = (path) => {
    if (path === '/dashboard') {
      return location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <aside className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
      <div className="p-6">
        <nav className="space-y-1">
          {menuItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 group ${
                  isActive(item.path)
                    ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-600 shadow-sm'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon className={`h-5 w-5 ${isActive(item.path) ? 'text-primary-600' : 'text-gray-400 group-hover:text-gray-600'}`} />
                <div className="flex-1">
                  <div className={`font-semibold text-sm ${isActive(item.path) ? 'text-primary-700' : 'text-gray-900'}`}>
                    {item.label}
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {item.description}
                  </div>
                </div>
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
