'use client';
import React, { useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

/** The site has one page: the LLM visualizer. Send visitors straight to it. */
export default function Page() {
    let router = useRouter();
    useEffect(() => { router.replace('/llm'); }, [router]);
    return <div className="p-4">
        <Link href="/llm" className="text-blue-700 underline">Open the LLM visualization</Link>
    </div>;
}
