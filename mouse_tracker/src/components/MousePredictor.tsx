import React, { useState, useEffect } from 'react';

interface MousePredictData {
    currentX: number;
    currentY: number;
    dx: number;
    dy: number;
    dt: number;
    d: number;
}

const MousePredictor: React.FC = () => {
    const [mouseData, setMouseData] = useState<MousePredictData[]>([]);
    const [lastMousePosition, setLastMousePosition] = useState<{ x: number; y: number } | null>(null);
    const [lastMouseTime, setLastMouseTime] = useState<number | null>(null);
    const [buttonPosition, setButtonPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
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
                { currentX, currentY, dx, dy, dt, d: 0 },
            ]);
            setLastMousePosition({ x: currentX, y: currentY });
            setLastMouseTime(currentTime);
        };
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [lastMousePosition, lastMouseTime]);

    useEffect(() => {
        if (mouseData.length > 0) {
            const lastData = mouseData[mouseData.length - 1];
            if (lastData && lastData.dt >= 500) {
                fetchPrediction();
            }
        }
    }, [mouseData]);

    const fetchPrediction = async () => {
        const recentData = mouseData.slice(-5).map(data => ({
            currentX: data.currentX,
            currentY: data.currentY,
            dx: data.dx,
            dy: data.dy,
            dt: data.dt,
            d: data.d,
        }));
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features: recentData.flatMap(data => [data.currentX, data.currentY, data.dx, data.dy, data.dt, data.d]) }),
        });
        const prediction = await response.json();
        console.log(prediction)
        setPredictedPosition({ x: prediction.targetX, y: prediction.targetY });
    };

    return (
        <div>
            {predictedPosition && (
                <div>
                    {predictedPosition.x}
                    {predictedPosition.y}
                <div
                    style={{
                        position: 'absolute',
                        left: `${predictedPosition.x}px`,
                        top: `${predictedPosition.y}px`,
                        width: '5px',
                        height: '5px',
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