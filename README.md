# STADIOchoice Subscriber Churn Prediction — Capstone Project (CAP182)

## Motivation

STADIOchoice is Africa's largest pay-TV and streaming company, with 21.5
million subscribers. The business is going through a risky shift: its
profitable satellite customers are dropping by about 5% every year, while
its streaming service, STADIOstream, is growing fast but still losing
money. This means every streaming subscriber the company already has is
very valuable — losing one wastes the money it took to sign them up in
the first place.

This problem affects several groups inside STADIOchoice: the retention
and customer service teams, who currently send discounts almost randomly
instead of to the right customers; the finance team, who watches
marketing budget get wasted on customers who were never going to leave;
and company leadership, who have publicly committed to winning through
retention as their top strategic priority.

Right now, STADIOchoice only reacts to churn — it does not predict it.
Streaming churn has climbed from 6.5% to 9.2% every month in just two
years, and spikes hard right after a big sports event or show ends. A
data-driven churn prediction model would let STADIOchoice see which
customers are at risk before they cancel, so it can send retention
offers only to customers actually likely to leave, save money currently
wasted on discounting customers who would have stayed anyway, and catch
"quiet leavers" who slowly stop watching but never contact support.

This is not a small or side issue for STADIOchoice — it is central to
their own stated 2030 strategy, which puts "win the streaming war on
retention, not just sign-ups" as priority number one. STADIOchoice
already collects three years of viewing data and five years of billing
data; what is missing is a model that turns this data into an early
warning system. This project delivers exactly that.

## Problem Statement

**Who is affected:** STADIOchoice's streaming service, STADIOstream, is
affected by a rising number of subscribers leaving the platform every
month. This affects the retention team, the finance team, and the wider
business, since streaming churn has grown from 6.5% to 9.2% per month in
two years.

**What is unknown:** STADIOchoice does not currently know, in advance,
which specific subscribers are likely to cancel their streaming
subscription. It cannot tell the difference between an at-risk
subscriber and one who was never going to leave, and it cannot detect
"quiet leavers" who stop watching but never contact support.

**What data will be used:** This project will use subscriber-level
behavioural and account data: streaming viewing logs, subscription and
billing history, and cancellation/retention records. As STADIOchoice's
real data is not yet accessible, this problem will first be investigated
using a comparable publicly available customer churn dataset, to prove
the approach is viable before applying it to STADIOchoice's own data.

## Repository Structure

This repository is organised as follows:

- **`datasets/`** — Raw and cleaned datasets used for this project,
  including the public churn dataset used as a proof of concept.
- **`models/`** — Saved/trained model files produced during the
  modelling stage (SS2 onward).
- **`experimental_setup/`** — Notebooks and scripts describing how
  experiments were configured (train/test splits, chosen algorithms,
  hyperparameters).
- **`experimental_results/`** — Output of model evaluations: metrics,
  confusion matrices, comparison results.
- **`scripts/statistical_helpers/`** — Reusable scripts for statistical
  calculations and model comparisons.
- **`scripts/visualisations/`** — Scripts used to generate charts and
  graphs for analysis and reporting.

## RAAIDD Log

### Risks
1. STADIOchoice's real subscriber data may not be provided in time, or
   at all, delaying the modelling stages in SS2.
2. The public dataset used as a proof of concept may not closely
   resemble STADIOchoice's real streaming data, which could limit how
   well the results transfer later.
3. Sensitive customer data (billing, viewing habits) could be misused
   or leaked if not handled and anonymised correctly.
4. The model may perform well on historical data but fail to generalise
   to new subscribers if churn patterns change over time (e.g. after a
   major content or price change).
5. Class imbalance in the churn data (far more subscribers stay than
   leave) could cause the model to under-predict churners if not
   addressed.

### Actions
1. Source and clean a public churn dataset now, so modelling work can
   start immediately without waiting on STADIOchoice's real data.
2. Set up the GitHub repository with the correct folder structure
   before beginning any coding work.
3. Draft a clear Data Request document (see `Data_Request.pdf`) so
   that, if real data becomes available, it is requested in a usable,
   well-structured format.
4. Regularly commit work to GitHub with clear messages, so progress is
   visible and traceable throughout the project.
5. Check techniques for handling class imbalance (e.g. resampling,
   class weighting) before starting the modelling stage in SS2.

### Assumptions
1. STADIOchoice's real subscriber data will eventually be made
   available in a similar structure to the one requested in the Data
   Request.
2. The public churn dataset used for the proof of concept is a
   reasonable stand-in for streaming subscriber behaviour, even though
   it comes from a different industry.
3. The churn prediction approach (supervised classification) will
   remain the most suitable method for this problem as the project
   develops.
4. STADIOchoice's business priorities (retention, reducing churn) will
   not change significantly during the course of this project.
5. The data provided will cover a long enough time period to capture
   meaningful churn patterns, including seasonal spikes after major
   events.

### Issues
Currently, there are no active issues. Since this is the proposal stage
(SS1), no real client data has been received yet, and no technical work
(data cleaning, modelling) has started — so no problems have arisen to
log at this point. This section will be updated as issues emerge during
SS2 and SS3.

### Decisions
A decision was made to use a publicly available customer churn dataset
as a proof-of-concept substitute for STADIOchoice's real data, since the
real data was not accessible at this stage. This allows modelling work
to begin without delay, in line with SS2's requirement to first prove
the approach on public data.

### Dependencies
1. Access to STADIOchoice's real subscriber data (viewing logs, billing
   history, cancellation records), once approved and shared.
2. Availability of a suitable public churn dataset with comparable
   features (subscription details, usage behaviour, churn outcome).
3. Access to Python and its data science libraries (e.g. pandas,
   scikit-learn) for data cleaning and modelling.
4. A working GitHub repository with lecturer access granted, since all
   submissions depend on this being set up correctly.
5. Feedback from the lecturer/marker on SS1, which may affect the
   direction or scope of work carried into SS2.

## Data Request

See [`Data_Request.pdf`](./Data_Request.pdf) for the full data request
submitted to the client.
