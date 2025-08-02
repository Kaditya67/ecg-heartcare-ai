// components/LazyLottie.jsx
import React, { useEffect, useRef, useState } from 'react';
import Lottie from 'lottie-react';

const LazyLottie = ({ animationData, className }) => {
  const ref = useRef();
  const [show, setShow] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShow(true);
          observer.disconnect();
        }
      },
      { threshold: 0.2 }
    );

    if (ref.current) observer.observe(ref.current);

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={className}>
      {show ? (
        <Lottie animationData={animationData} loop autoplay />
      ) : (
        <div className="w-full h-full bg-gradient-to-br from-gray-300 to-gray-100 dark:from-gray-700 dark:to-gray-800 animate-pulse rounded-xl" />
      )}
    </div>
  );
};

export default LazyLottie;
