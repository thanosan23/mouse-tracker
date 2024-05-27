import React, { useState, useEffect } from 'react';

interface MousePredictData {
    currentX: number;
    currentY: number;
    dx: number;
    dy: number;
    dt: number;
}

const DEBUG = false;
const HOST = 'localhost';
const PORT = 8765;
const WINDOW_SIZE = 10;

const MousePredictor: React.FC = () => {
    const [mouseData, setMouseData] = useState<MousePredictData[]>([]);
    const [lastMousePosition, setLastMousePosition] = useState<{ x: number; y: number } | null>(null);
    const [lastMouseTime, setLastMouseTime] = useState<number | null>(null);
    const [predictedPosition, setPredictedPosition] = useState<{ x: number; y: number } | null>(null);

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
                { currentX, currentY, dx, dy, dt },
            ]);
            setLastMousePosition({ x: currentX, y: currentY });
            setLastMouseTime(currentTime);
        };
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [lastMousePosition, lastMouseTime]);

    useEffect(() => {
        if (mouseData.length > WINDOW_SIZE) {
            const lastData = mouseData[mouseData.length - 1];
            if (lastData) {
                fetchPrediction();
            }
        }
    }, [mouseData]);

    const fetchPrediction = async () => {
        const recentData = mouseData.slice(-WINDOW_SIZE).map(data => ({
            currentX: data.currentX,
            currentY: data.currentY,
            dx: data.dx,
            dy: data.dy,
            dt: data.dt,
        }));

        const ws = new WebSocket(`ws://${HOST}:${PORT}/predict`);

        ws.onopen = () => {
            ws.send(
                JSON.stringify({ features: recentData.flatMap(data => [data.currentX, data.currentY, data.dx, data.dy, data.dt] )})
            );
        };

        ws.onmessage = (event) => {
            const prediction = JSON.parse(event.data);
            setPredictedPosition({ x: prediction.targetX, y: prediction.targetY });
            ws.close();
        };
    };

    return (
        <div>
            {predictedPosition && (
                <div>
                    { DEBUG && (
                        <div className="text-xs">
                            <p>Debug</p>
                            <p>Predicted: {predictedPosition.x} {predictedPosition.y}</p>
                        </div>
                    ) }
                    <div
                        style={{
                            position: 'absolute',
                            left: `${predictedPosition.x}px`,
                            top: `${predictedPosition.y}px`,
                            width: '10px',
                            height: '10px',
                            backgroundColor: 'rgba(255, 0, 0, 0.5)',
                            border: '2px solid red',
                            borderRadius: '10%',
                        }}
                    />
                </div>
            )}
        </div>
    );
};
export default MousePredictor;
