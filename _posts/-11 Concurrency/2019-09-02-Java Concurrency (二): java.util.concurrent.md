---
layout: post
title: Java Concurrency (二)：java.util.concurrent
categories: [-11 Concurrency]
tags: [Concurrency]
number: [-11.02]
fullview: false
shortinfo: Java java.util.conurrent库的总结。

---
目录
{:.article_content_title}


* TOC
{:toc}

---
{:.hr-short-left}

## 1. 分工

### 1.1 Excutor与线程池

### 1.2 Fork/Join

> Fork/Join: 多线程的divide and conqure思想。`RecursiveTask`递归定义了任务如何divide和conqure， 实例方法`fork`和`join`定义了多线程的如何协作。 `ForkJoinPool`管理`RecursiveTask`的线程池。Fork/Join类似与单机版的Map/Reduce。

{% highlight java linenos %}

package divide;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveTask;

public class ForkAndJoinAddition extends RecursiveTask<Long> {

  final long THRESH_HOLD = 10000000000L;

  long start;
  long end;
  long sum;

  ForkAndJoinAddition(long start, long end) {
    this.start = start;
    this.end = end;
    this.sum = 0;
  }

  @Override
  protected Long compute() {
   
   Boolean shouldCompute = end - start < this.THRESH_HOLD; 

   if(shouldCompute) {
      for (long i = this.start; i <= this.end; i++) {
        this.sum += i;
      }
   } else {
      long middle = (start + end) / 2;
      ForkAndJoinAddition left = new ForkAndJoinAddition(start, middle);
      ForkAndJoinAddition right = new ForkAndJoinAddition(middle + 1, end);

      left.fork();
      right.fork();

      long leftSum = left.join();
      long rightSum = right.join();

      this.sum = leftSum + rightSum;
   }

    return this.sum;
  }

  public static long run(long start, long end) {
    ForkJoinPool forkJoinPool = new ForkJoinPool();
    ForkAndJoinAddition forkAndJoinAddition = new ForkAndJoinAddition(start, end);
    Future<Long> totalSum = forkJoinPool.submit(forkAndJoinAddition);

    try {
      long result = totalSum.get();
      System.out.println(start + " + ... + " + end + " = " + result);
      return result;
    } catch(Exception e) {
      return 0;
    }
  }
  public static void main(String[] args) {
    ForkAndJoinAddition.run(1,4);
  }
}

{% endhighlight %}

### 1.3 Future

> Future, 类似与js的Promise，封装了一个异步过程的返回值或错误。通常与`Callable`结合(`Future<Integer> result = executor.submit(task)`)。

{% highlight java linenos %}
public class Test {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newCachedThreadPool();
        Task task = new Task();
        Future<Integer> result = executor.submit(task);
        executor.shutdown();
         
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e1) {
            e1.printStackTrace();
        }
         
        System.out.println("主线程在执行任务");
         
        try {
            System.out.println("task运行结果"+result.get());
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
         
        System.out.println("所有任务执行完毕");
    }
}
class Task implements Callable<Integer>{
    @Override
    public Integer call() throws Exception {
        System.out.println("子线程在进行计算");
        Thread.sleep(3000);
        int sum = 0;
        for(int i=0;i<100;i++)
            sum += i;
        return sum;
    }
}
{% endhighlight %}

### 1.4 Guarded Suspension模式

> Guarede suspension: 获取锁之后需要检查某一条件是否满足，若不满足，则block，直到满足。满足后需要重新检测条件，通过while而不是if来实现。

{% highlight java linenos %}


public class Example {
    synchronized void guardedMethod() {
        while (!preCondition()) {
            try {
                // Continue to wait
                wait();
                // …
            } catch (InterruptedException e) {
                // …
            }
        }
        // Actual task implementation
    }
    synchronized void alterObjectStateMethod() {
        // Change the object state
        // …
        // Inform waiting threads
        notify();
    }
}

{% endhighlight %}

### 1.5 Balking模式

> Balking: 获取锁后，若条件不满足，立即返回(不像Guarded Suspension，一直等待直到条件满足)。 Balking模式适用与两个线程做同样的动作，若其中一个做完，另一个不做立即返回。比如有个用户线程要手动存储修改的数据到文件，还有个后台线程以每秒一次的频率自动存储数据到文本， 若用户线程已经在1秒的间隔内手动保存，则后台线程1秒到后检查到已存储，则立即返回；若用户线程修改后在1秒的间隔内不存储，则后台线程1秒到后检查到为存储，则执行存储。此后用户线程超常进行存储时，发现修改已存储，则不存储。

{% highlight java linenos %}

// Data.java

package divide;

import java.io.FileWriter;
import java.io.IOException;

public class Data {
  private String fileName;
  private String content;
  private Boolean changed;

  Data(String fileName, String content) {
    this.fileName = fileName;
    this.content = content;
    this.changed = false;
  }

  synchronized void change(String content) {
    this.content = content;
    this.changed = true;
  }

  synchronized void save() throws IOException {
    if (!this.changed) {
      return;
    }
    this.doSave();
  }

  void doSave() throws IOException {
    System.out.println(Thread.currentThread().getName() + " calls doSave, content = " + content);
    FileWriter fileWrite = new FileWriter(this.fileName);
    fileWrite.write(this.content);
    fileWrite.close();
    this.changed = false;
  }
  
}

// ChangerThread.java
package divide;

import java.io.IOException;
import java.util.Random;

class ChangerThread extends Thread {
  private final Data data;
  private final Random random = new Random();

 ChangerThread(String name, Data data) {
   super(name);
   this.data = data;
 }

  public void run() {
    try {
      for (int i = 0; true; i++) {
        this.data.change("No: " + i);
        Thread.sleep(random.nextInt(1000)); //执行其他操作
        this.data.save();
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch(InterruptedException e) {
      e.printStackTrace();
    }
  }
}

// SaverThread

package divide;

import java.io.IOException;

class SaverThread extends Thread {
  private final Data data;

 SaverThread(String name, Data data) {
   super(name);
   this.data = data;
 }

  public void run() {
    try {
      while (true) {
        this.data.save();
        Thread.sleep(1000);
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch(InterruptedException e) {
      e.printStackTrace();
    }
  }
}

// Balking.java

public class Balking {
  public static void main(String[] args) {
    System.out.println("Test Balking");
    Data data = new Data("data.txt", "empty");
    new ChangerThread("ChangerThread", data).start();
    new SaverThread("SaverThread", data).start();
  }
}

{% endhighlight %}


### 1.6 Thread-Per-Message模式

> Thread-Per-Message: 每一个请求都会被在主线程里分配一个子线程，然后主线程立即返回。类似于socket建立连接时，服务端的监听socket和处理socket(只不过处理socket的策略可以是多线程或者是单线程策略，例如select, poll)。

{% highlight java linenos %}

package divide;

public class ThreadPerMessage {
  private Worker worker;

  ThreadPerMessage() {
    this.worker = new Worker();
  }

  public void handle(int count, String message) {
   new Thread() {
     public void run() {
       worker.handle(count, message);
     }
   }.start(); 
  }

  public static void main(String[] args) {
    ThreadPerMessage threadPerMessage = new ThreadPerMessage();
    threadPerMessage.handle(10, "A");
    threadPerMessage.handle(20, "B");
    threadPerMessage.handle(30, "C");
    System.out.println("Finished");
  }
}

class Worker {
  public void handle(int count, String message) {
    String threadName = Thread.currentThread().getName();
    System.out.println(threadName +  "        handle(" + count + ", " + message + ") BEGIN");
    for (int i = 0; i < count; i++) {
      System.out.println(threadName + ": " + i + ": " + message);
      this.slowly();
    }
    System.out.println(threadName + "        handle(" + count + ", " + message + ") END");
  }
  private void slowly() {
    try {
        Thread.sleep(100);
    } catch (InterruptedException e) {
    }
  }
}

{% endhighlight %}

### 1.7 生产者-消费者模式

> Producer-Consumer: 3点需要考虑。1）生产者只能在队列不满的时候生产，否则block, 2)消费者只能在队列不空的时候消费，否则block。3）多个生产者之间，和多个消费者之间的race condition要解决。

以下是Semaphore实现的生产者消费者队列。

{% highlight java linenos %}
package BoundedBufferConcurrent;

import java.util.concurrent.Semaphore;

import net.jcip.annotations.GuardedBy;
import net.jcip.annotations.ThreadSafe;;

@ThreadSafe
public class SemaphoreBoundedBuffer<E> {
  private final Semaphore availableItems, availableSpaces;
  @GuardedBy("this")
  private final E[] items;
  @GuardedBy("this")
  private int putPosition = 0, takePosition = 0;

  public SemaphoreBoundedBuffer(int capacity) {
    if (capacity <= 0)
      throw new IllegalArgumentException();
    availableItems = new Semaphore(0);
    availableSpaces = new Semaphore(capacity);
    items = (E[]) new Object[capacity];
  }

  public boolean isEmpty() {
    return availableItems.availablePermits() == 0;
  }

  public boolean isFull() {
    return availableSpaces.availablePermits() == 0;
  }

  public void put(E x) throws InterruptedException {  // <====== point 1
    availableSpaces.acquire();
    doInsert(x);
    availableItems.release();
  }

  public E take() throws InterruptedException {   // <====== point 2
    availableItems.acquire();
    E item = doExtract();
    availableSpaces.release();
    return item;
  }

  private synchronized void doInsert(E x) { // <====== point 3
    int i = putPosition;
    items[i] = x;
    putPosition = (++i == items.length) ? 0 : i;
  }

  private synchronized E doExtract() {  // <====== point 3
    int i = takePosition;
    E x = items[i];
    items[i] = null;
    takePosition = (++i == items.length) ? 0 : i;
    return x;
  }
}

{% endhighlight %}

### 1.8 Worker Thread模式

> Worker Thread: 类似与Producer和Consumer模式中的Consumer，只不过Worker Thread更加被动，一旦array不为空，自动consume, 完成后等待array里的下一个request。而Producer和Consumer模式中的Consumer是主动的，不会永远等待消费，消费完就结束了。Worker Thread模式也被成为Background Thread（背景线程）模式，另外，如果从保存多个工人线程的场所这一点看，我们也可以称这种模式为Thread Pool模式

{% highlight java linenos %}
public class WorkerThread extends Thread {

    private static final Random random = new Random(System.currentTimeMillis());
    private final Channel channel;

    public WorkerThread(String name, Channel channel) {
        super(name);
        this.channel = channel;
    }

    @Override
    public void run() {
        while (true) {
            channel.take().execute();

            try {
                Thread.sleep(random.nextInt(1_000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
{% endhighlight %}

### 1.9 两阶段终止模式

> Two Phase Termination: 主要解决在线程工作周期中间不要立即中断，否则会引起不可预料后果，比如resource没有释放(db connection)。 通过在每个周期开始时检查是否终止来决定进行业务逻辑或清理逻辑。

{% highlight java linenos %}

package divide;

class TwoPhaseTermination extends Thread {
  private boolean isShutdown;
  private long counter;

  TwoPhaseTermination() {
    this.isShutdown = false;
    this.counter = 0;
  }

  public void setIsShutdown(boolean isShutdown) {
    this.isShutdown = isShutdown;
  }

  public boolean getIsShutDown() {
    return this.isShutdown;
  }

  public void run() {
    try {
      while (!isShutdown) {
        doWork();
      }
    } catch (Exception e) {

    } finally {
      this.doClean();
    }
  }

  void doWork() {
    System.out.println(Thread.currentThread().getName() + " do work: " + counter);
    try {
      Thread.sleep(500);
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    counter++;
  }

  void doClean() {
    System.out.println("cleaning during shutdown");

  }

  public static void main(String[] args) {
    try {
      TwoPhaseTermination tpt = new TwoPhaseTermination();
      tpt.start();
      Thread.sleep(10000);
      tpt.setIsShutdown(true);
      System.out.println("ask Shutdown thread to shutdown");
      tpt.join();
      System.out.println("Shutdown thread finished");
    } catch (Exception e) {


    }
  }
}

{% endhighlight %}

## 2 协作(sleep-awake)

### 2.1 信号量(Semaphore)

[See 1.7 生产者消费者模式]({{site.url}}/-11 concurrency/2019/09/02/Java-Concurrency-(二)-java.util.concurrent.html#17-生产者-消费者模式)

### 2.2 管程(Monitor)：Lock & Condition + Synchronized

[See synchronization]({{site.url}}/-11 concurrency/2019/09/01/Java-Concurrency-(一)-Thread.html#6-synchronization)

### 2.3 倒数计数锁(CountDownLatch)

> CountDownLatch: Java多线程编程中经常会碰到这样一种场景——某个线程需要等待一个或多个线程操作结束（或达到某种状态）才开始执行。比如开发一个并发测试工具时，主线程需要等到所有测试线程均执行完成再开始统计总共耗费的时间，此时可以通过CountDownLatch轻松实现。

{% highlight java linenos %}

package coordination;

import java.util.concurrent.CountDownLatch;

class CountDownLatchApp {
  CountDownLatchApp(){
  }

  public static void main(String[] args) throws InterruptedException {

    int totalThreads = 3;
    CountDownLatch countDownLatch = new CountDownLatch(totalThreads);

    long start = System.currentTimeMillis();
    for (int i = 0; i < totalThreads; i++) {
      Thread thread = new Thread() {
        public void run() {
          try {
            System.out.println(Thread.currentThread().getName() + " start");
            Thread.sleep(500);
            countDownLatch.countDown();
          } catch (Exception e) {

          }
        }
      };
      thread.start();
    }
    countDownLatch.await();
    long end = System.currentTimeMillis();
    System.out.println("total takes: " + (end - start)); 
  }
}

{% endhighlight %}

可以看到，主线程等待所有3个线程都执行结束后才开始执行。

### 2.4 回环栏栅(CyclicBarrier)

> CyclicBarrier: 。CyclicBarrier可以译为循环屏障，保证屏障之前的代码一定在屏障之后的代码之前被执行。CyclicBarrier可以在构造时指定需要在屏障前执行await的个数，所有对await的调用都会等待，直到调用await的次数达到预定指，所有等待都会立即被唤醒

{% highlight java linenos %}
package coordination;

import java.util.concurrent.CyclicBarrier;

class CyclicBarrierApp {

  public static void main(String[] args) throws InterruptedException {

    int totalThreads = 3;
    CyclicBarrier cyclicBarrier = new CyclicBarrier(totalThreads);

    for (int i = 0; i < totalThreads; i++) {
      Thread thread = new Thread() {
        public void run() {
          try {
            System.out.println(Thread.currentThread().getName() + " start");
            Thread.sleep(500);
            cyclicBarrier.await();
            System.out.println(Thread.currentThread().getName() + " end");
          } catch (Exception e) {

          }
        }
      };
      thread.start();
    }
  }
}
{% endhighlight %}

> CountDownLatch vs CyclicBarrier: 从使用场景上来说，CyclicBarrier是让多个线程互相等待某一事件的发生，然后同时被唤醒。而上文讲的CountDownLatch是让某一线程等待多个线程的状态，然后该线程被唤醒。

### 2.5 Phaser

> Phaser顾名思义，与阶段相关。Phaser比较适合这样一种场景，一种任务可以分为多个阶段，现希望多个线程去处理该批任务，对于每个阶段，多个线程可以并发进行，但是希望保证只有前面一个阶段的任务完成之后才能开始后面的任务。这种场景可以使用多个CyclicBarrier来实现，每个CyclicBarrier负责等待一个阶段的任务全部完成。但是使用CyclicBarrier的缺点在于，需要明确知道总共有多少个阶段，同时并行的任务数需要提前预定义好，且无法动态修改。而Phaser可同时解决这两个问题。

{% highlight java linenos %}

package coordination;

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Phaser;

class PhaserApp {

  public static void main(String[] args) throws InterruptedException {

    int totalThreads = 3;
    int phases = 5;
    Phaser phaser = new Phaser(totalThreads) {
      @Override
      protected boolean onAdvance(int phase, int registeredParties) {
        System.out.println("==== phase: " + phase);
        return super.onAdvance(phase, registeredParties);
      }
    };

    for (int i = 0; i < totalThreads; i++) {
      Thread thread = new Thread() {
        public void run() {

          for (int i = 0; i < phases; i++) {
            try {
              System.out.println(Thread.currentThread().getName() + " start phase: " + phases);
              Thread.sleep(500);
            } catch (Exception e) {
  
            }
            phaser.arriveAndAwaitAdvance();
          }
        }
      };
      thread.start();
    }
  }
}

/*
Output:
Thread-1 start phase: 5
Thread-0 start phase: 5
Thread-2 start phase: 5
==== phase: 0
Thread-0 start phase: 5
Thread-2 start phase: 5
Thread-1 start phase: 5
==== phase: 1
Thread-0 start phase: 5
Thread-2 start phase: 5
Thread-1 start phase: 5
==== phase: 2
Thread-1 start phase: 5
Thread-0 start phase: 5
Thread-2 start phase: 5
==== phase: 3
Thread-1 start phase: 5
Thread-2 start phase: 5
Thread-0 start phase: 5
==== phase: 4

*/

{% endhighlight %}

### 2.5 Exchanger

> Exchanger: Exchanger可以在两个线程之间交换数据，只能是2个线程，他不支持更多的线程之间互换数据。当线程A调用Exchange对象的exchange()方法后，他会陷入阻塞状态，直到线程B也调用了exchange()方法，然后以线程安全的方式交换数据，之后线程A和B继续运行

{% highlight java linenos %}
package coordination;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Exchanger;
import java.util.concurrent.Phaser;

class ExchangerApp {

  public static void main(String[] args) throws InterruptedException {
    Exchanger exchanger = new Exchanger<ArrayList<Integer>>();
    Producer producer = new Producer(exchanger);
    Consumer consumer = new Consumer(exchanger);
    producer.start();
    consumer.start();
  }
}

class Producer extends Thread {
  Exchanger exchanger;
  ArrayList<Integer> list;

  Producer(Exchanger exchanger) {
    this.exchanger = exchanger;
    this.list = new ArrayList<Integer>();
  }

  public void run() {
    for (int i = 0; i < 3; i++) {
      this.list.add(i);
      this.list.add(i);
      this.list.add(i);
      try {
        this.exchanger.exchange(this.list);
        for (Integer temp : this.list) {
          System.out.println("From producer: " + temp);
        }
        this.list = new ArrayList<Integer>();
      } catch (Exception e) {

      }
    }
  }
}

class Consumer extends Thread {
  Exchanger exchanger;
  ArrayList<Integer> list;

  Consumer(Exchanger exchanger) {
    this.exchanger = exchanger;
    this.list = new ArrayList<Integer>();
  }

  public void run() {
    for (int i = 0; i < 3; i++) {
      int j = i * 10;
      this.list.add(j);
      this.list.add(j);
      this.list.add(j);
      try {
        this.exchanger.exchange(this.list);
        for (Integer temp : this.list) {
          System.out.println("From consumer: " + temp);
        }
        this.list = new ArrayList<Integer>();

      } catch (Exception e) {

      }
    }
  }
}
{% endhighlight %}

## 3 互斥(Mutex)

### 3.1 with lock

#### 3.1.1 synchrnozed

[See synchronization]({{site.url}}/-11 concurrency/2019/09/01/Java-Concurrency-(一)-Thread.html#6-synchronization

#### 3.1.2 Lock

[See synchronization]({{site.url}}/-11 concurrency/2019/09/01/Java-Concurrency-(一)-Thread.html#6-synchronization

#### 3.1.3 Write/Read Lock

> Write/Read lock, 是为了解决普通锁读与读之间的互斥可以优化的问题，使用Write/Read lock后，读锁与读锁之间不互斥，读锁与写锁之间互斥，写锁与写锁之间互斥。

{% highlight java linenos %}
package mutex;

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Hello world!
 */
public final class ReadWriteApp {
    /**
     * Says hello to the world.
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
      Data data = new Data();

      for(int i = 0; i < 3; i++) {
        new Thread() {
          public void run() {
            data.get();
          }
        }.start();;
      }

      for(int i = 0; i < 3; i++) {
        int j = i;
        new Thread() {
          public void run() {
            data.put(j);
          }
        }.start();;
      }
    }
}


class Data {
  private Object data = null;
  private ReentrantReadWriteLock rwl = null;

  Data() {
    this.data = "A";
    this.rwl = new ReentrantReadWriteLock(false);
  }

  public void get() {
    rwl.readLock().lock();
    try {
      System.out.println(Thread.currentThread().getName() + " GET: be ready to read data!");
      Thread.sleep(5000);
      System.out.println(Thread.currentThread().getName() + " GET: finished read data: " + this.data);
    } catch( Exception e) {

    }
    rwl.readLock().unlock();
  }

  public void put(Object object) {
    rwl.writeLock().lock();
    try {
      System.out.println(Thread.currentThread().getName() + " PUT: be ready to put data!");
      Thread.sleep(5000);
      this.data = object;
      System.out.println(Thread.currentThread().getName() + " PUT: finished put data: " + this.data);
    } catch( Exception e) {

    }
    rwl.writeLock().unlock();
  }
}

/* output
Thread-2 GET: be ready to read data!
Thread-1 GET: be ready to read data!
Thread-0 GET: be ready to read data!
Thread-2 GET: finished read data: A
Thread-1 GET: finished read data: A
Thread-0 GET: finished read data: A
Thread-3 PUT: be ready to put data!
Thread-3 PUT: finished put data: 0
Thread-4 PUT: be ready to put data!
Thread-4 PUT: finished put data: 1
Thread-5 PUT: be ready to put data!
Thread-5 PUT: finished put data: 2
*/

{% endhighlight %}

### 3.2 without lock

{% highlight java linenos %}

{% endhighlight %}

#### 3.2.1 不变模式(Immutable)

#### 3.2.2 线程本地存储(ThreadLocal)

> ThreadLocal: 提供了一个线程本地可见的副本，线程之间无法查看其它线程的变量，起到了隔离的作用。

{% highlight java linenos %}
package mutex;

public final class ThreadLocalApp {

  public static void main(String[] args) {

    MyRunnable myRunnable = new MyRunnable();

    Thread threadA = new Thread(myRunnable);
    Thread threadB = new Thread(myRunnable);

    threadA.start();
    threadB.start();
 
  }
}

class MyRunnable implements Runnable {
  private ThreadLocal threadLocal = new ThreadLocal() {
    protected String initialValue() {
      return Thread.currentThread().getName();
    }
      
  };

  public void run() {
    try {
      System.out.println(Thread.currentThread().getName() + " start set" + threadLocal.get());
      threadLocal.set(Double.toString(Math.random()));
      System.out.println(Thread.currentThread().getName() + " finish set: " + threadLocal.get());
      Thread.sleep(1000);
      System.out.println(Thread.currentThread().getName() + " finish set 2: " + threadLocal.get());
    } catch( Exception e) {

    }
  }
}

/* output
Thread-1 start setThread-1
Thread-0 start setThread-0
Thread-1 finish set: 0.6876997267669848
Thread-0 finish set: 0.5453606964177355
Thread-1 finish set 2: 0.6876997267669848
Thread-0 finish set 2: 0.5453606964177355
*/

可以看到同一个MyRunnable实例myRunnalbe在两个线程里启动，但是myRunnable的实例变量threadLocal在不同线程的初始值是不一样的，在两个线程都写之后，再读取到的值也是相互独立的。

{% endhighlight %}

#### 3.2.3 CAS

> CAS(CompareAndSwap): 当前值等于预期值时(即变量没有在上次读取和这次代码执行之间没有被改变)，则设置新值，返回true；否则什么也不做，返回false

{% highlight java linenos %}
public class SimulateCAS {
    private int value;
    public synchronized int getValue() {
        return value;
    }
    public boolean compareAndSet(int expect, int newValue) {
        synchronized (this) {
            if (value == expect) {
                value = newValue;
                return true;
            }
        }
        return false;
    }
}
{% endhighlight %}


#### 3.2.4 Copy-on-Write

> Copy-on-Write(COW): 当我们往一个容器添加元素的时候，不直接往当前容器添加，而是先将当前容器进行Copy，复制出一个新的容器，然后新的容器里添加元素，添加完元素之后，再将原容器的引用指向新的容器。
这样做的好处是我们可以对CopyOnWrite容器进行并发的读，而不需要加锁，因为当前容器不会添加任何元素。

{% highlight java linenos %}

{% endhighlight %}


#### 3.2.5 原子类

> `java.util.concurrent.atomic`包括了如下原子类: AtomicBoolean，AtomicInteger，AtomicLong，AtomicReference
AtomicIntegerArray，AtomicLongArray
AtomicLongFieldUpdater，AtomicIntegerFieldUpdater，AtomicReferenceFieldUpdater
AtomicMarkableReference，AtomicStampedReference，AtomicReferenceArray

{% highlight java linenos %}

{% endhighlight %}



## 3 总结 ##

{: .img_middle_hg}
![regular expression]({{site.url}}/assets/images/posts//-11_Concurrency/2019-09-01-Java Concurrency Foundamentals/Java thread state machine.png)



{% highlight java linenos %}

{% endhighlight %}

## 4 参考资料 ##







