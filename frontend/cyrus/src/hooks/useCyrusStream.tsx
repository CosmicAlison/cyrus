import { useEffect, useState } from "react";

export function useCyrusStream(jobId: string, api_url) {
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    const source = new EventSource(
      `${api_url}?job_id=${jobId}`
    );

    source.onmessage = (event) => {
        const payload = JSON.parse(event.data);

        setEvents((prev) => [
            ...prev,
            payload
        ]);
    };

    return () => source.close();

  }, [jobId]);

  return events;
}