import React from 'react';

const ProbabilityBar = ({ probabilities, highlight }) => {
  const colors = {
    cold: 'bg-gray-400',
    warm: 'bg-yellow-400',
    hot: 'bg-red-400'
  };

  const highlightColors = {
    cold: 'bg-gray-600',
    warm: 'bg-yellow-600',
    hot: 'bg-red-600'
  };

  return (
    <div className="space-y-1">
      {/* Probability Bar */}
      <div className="flex h-2 bg-gray-200 rounded-full overflow-hidden">
        {Object.entries(probabilities).map(([class_name, prob]) => (
          <div
            key={class_name}
            className={`h-full transition-all duration-300 ${
              class_name === highlight 
                ? highlightColors[class_name] 
                : colors[class_name]
            }`}
            style={{ width: `${prob * 100}%` }}
          />
        ))}
      </div>
      
      {/* Labels */}
      <div className="flex justify-between text-xs text-gray-500">
        {Object.entries(probabilities).map(([class_name, prob]) => (
          <span 
            key={class_name}
            className={`font-medium ${
              class_name === highlight ? 'text-gray-900' : 'text-gray-500'
            }`}
          >
            {class_name}: {(prob * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
};

export default ProbabilityBar;
