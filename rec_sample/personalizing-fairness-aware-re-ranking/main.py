from download import download_movielens
from data import load_ratings, load_movies
from preprocess import preprocess_movie
from recommendation.knn import KNNRecommender
from recommendation.svd import SVDRecommender
from rerank.diversity import DiversityReranker
from evaluate import coverage


def main() -> None:
    # download dataset
    download_movielens()

    # load dataset
    rating_df = load_ratings()
    print(rating_df.head())
    print(rating_df.shape)

    movie_df = load_movies()
    print(movie_df.head())
    print(movie_df.shape)

    # preprocess movies
    movie_df = preprocess_movie(movie_df)
    print(movie_df.head())
    print(movie_df.shape)

    # compute KNN recommender
    recommender = KNNRecommender()
    recommender.compute(rating_df)
    print("KNN recommender computed.")

    # compute Diversity Reranker
    reranker = DiversityReranker()
    reranker.compute(rating_df, movie_df)
    print("Diversity Reranker computed.")
    print()

    # recommendation + reranking
    user_id = 10
    movie_ids, movie_scores = recommender.score_items(user_id, top_n=50)
    movie2provider = dict(zip(movie_df['movie_id'], movie_df['provider']))

    reranked_movie_ids = reranker.rerank(user_id, movie_ids, movie_scores, movie2provider, lambda_=0.5, top_n=10)

    # evaluate
    coverage_score = coverage(reranked_movie_ids, movie2provider)

    print("KNN + Diversity Reranking Results")
    print("user_id:", user_id)
    print("Original movie IDs:", movie_ids[:10])
    print("Reranked movie IDs:", reranked_movie_ids)
    print("Coverage score after reranking:", coverage_score)
    print()
    del recommender

    # compute SVD recommender
    recommender = SVDRecommender()
    recommender.compute(rating_df, k=20)
    print("SVD recommender computed.")
    print()

    # recommendation + reranking
    user_id = 10
    movie_ids, movie_scores = recommender.score_items(user_id, top_n=20)
    movie2provider = dict(zip(movie_df['movie_id'], movie_df['provider']))
    reranked_movie_ids = reranker.rerank(user_id, movie_ids, movie_scores, movie2provider, lambda_=0.5, top_n=10)

    # evaluate
    coverage_score = coverage(reranked_movie_ids, movie2provider)

    print("SVD + Diversity Reranking Results")
    print("user_id:", user_id)
    print("Original movie IDs:", movie_ids[:10])
    print("Reranked movie IDs:", reranked_movie_ids)
    print("Coverage score after reranking:", coverage_score)
    del recommender

    print("DONE")


if __name__ == "__main__":
    main()
