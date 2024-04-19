import Head from "next/head";
import Link from "next/link";
import MouseTracker from "../components/MouseTracker"

export default function Home() {
  return (
    <>
      <Head>
        <title>Mouse Tracker</title>
        <meta name="description" content="Generated by create-t3-app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="">
        <MouseTracker />
      </main>
    </>
  );
}
