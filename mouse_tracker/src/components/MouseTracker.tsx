import React, { useState, useEffect } from 'react';

interface MouseData {
    currentX: number;
    currentY: number;
    dx: number;
    dy: number;
    dt: number;
    d: number;
    targetX: number | null;
    targetY: number | null;
}

const MouseTracker: React.FC = () => {
    const [mouseData, setMouseData] = useState<MouseData[]>([]);
    const [lastMousePosition, setLastMousePosition] = useState<{ x: number; y: number } | null>(null);
    const [lastMouseTime, setLastMouseTime] = useState<number | null>(null);
    const [buttonPosition, setButtonPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });


    useEffect(() => {
        console.log(mouseData);
    }, [mouseData]);

    useEffect(() => {
        const handleMouseMove = (event: MouseEvent) => {
            const currentTime = Date.now();
            const currentX = event.clientX;
            const currentY = event.clientY;
            const dx = currentX - (lastMousePosition ? lastMousePosition.x : currentX);
            const dy = currentY - (lastMousePosition ? lastMousePosition.y : currentY);
            const dt = lastMouseTime ? currentTime - lastMouseTime : 0;
            setMouseData((prevData) => [
                ...prevData,
                { currentX, currentY, dx, dy, dt, d: 0, targetX: null, targetY: null },
            ]);
            setLastMousePosition({ x: currentX, y: currentY });
            setLastMouseTime(currentTime);
        };
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [lastMousePosition, lastMouseTime]);

    useEffect(() => {
        positionButton();
    }, []);

    const positionButton = () => {
        const x = Math.random() * (window.innerWidth - 50); // 50 is button width
        const y = Math.random() * (window.innerHeight - 50); // 50 is button height
        setButtonPosition({ x, y });
    };

    const handleButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        const targetX = event.clientX;
        const targetY = event.clientY;

        // Update the last data point with target position and distance (d)
        setMouseData((prevData) =>
                     prevData.map((data, index) =>
                                  data.targetX == null
                                      ? {
                                          ...data,
                                          targetX,
                                          targetY,
                                          d: Math.sqrt(Math.pow(targetX - data.dx, 2) + Math.pow(targetY - data.dy, 2)),
                                      }
                                          : data,
                                 ),
                    );

                    positionButton();
    };

    const downloadCSV = () => {
        const csvContent =
            'currentX,currentY,dx,dy,dt,d,targetX,targetY\n' +
            mouseData
        .filter((data) => data.targetX != null)
        .map((data) => `${data.currentX},${data.currentY},${data.dx},${data.dy},${data.dt},${data.d},${data.targetX},${data.targetY}`)
        .join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'mouse_data.csv';
        link.click();
    };

    return (
        <div>
            <button
                style={{ position: 'absolute', left: `${buttonPosition.x}px`, top: `${buttonPosition.y}px` }}
                onClick={handleButtonClick}
            >
                Click Me
            </button>
            <button onClick={downloadCSV} style={{ position: 'fixed', bottom: '20px', right: '20px' }}>
                Download CSV
            </button>
        </div>
    );
};

export default MouseTracker;
